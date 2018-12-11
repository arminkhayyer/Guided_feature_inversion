import os
import copy
import pandas as pd
import numpy as np
import cv2
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms
from PIL import Image
import torch.optim
import matplotlib.cm as mpl_color_map
from guided_feature_inversion_convNN import Vgg19, preprocess_image, find_gussian_blur, recreate_image
import gzip
from xml.dom import minidom
import xml.etree.ElementTree as ET
import io
import requests
import skimage
import torchvision.transforms.functional as F
import tarfile
import imageio
import urllib.request



def preprocess_image_eval(image):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    transform = transforms.Compose([
        #transforms.Scale(size=256),
        transforms.Resize(size=[224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    img = transform(image).unsqueeze(0)
    #img = torch.from_numpy(img).float()

    return img


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

class Disciminative_vgg19():

    def __init__(self):
        super(Disciminative_vgg19, self).__init__()
        self.model= models.vgg19(pretrained=True)

        self.model.eval()
        self.select_layers= {"feature_inversion":36, "base_layer":35}
        if not os.path.exists('generated'):
            os.makedirs('generated')


    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = []
        for name, layer in enumerate(self.model.features):
            #print("layer", self.model.classifier(x))
            x = layer(x)
            if name in self.select_layers.values():
                features.append(x)
        return features

    def create_m_mask(self, weights, input_image):
        # defining the m mask
        base_layer = self.forward(input_image)[0]

        m_mask = torch.zeros(14, 14)
        for i, key in enumerate(base_layer):

            for j, b in enumerate(key):
                m_mask += weights[j] * b
        # min-max-scaling

        min_m = torch.min(m_mask)
        max_m = torch.max(m_mask)
        Range = max_m - min_m
        m_mask = (m_mask - min_m) / Range

        # upsamling
        m_mask = m_mask.view(1, 1, 14, 14)
        upsample = nn.Upsample(size=(224, 224), mode="bilinear")
        m_mask = upsample(m_mask)
        m_mask = torch.cat((m_mask, m_mask, m_mask), 1)

        return m_mask

    def discriminative_mask_and_weights(self, input_image, gussian_blur):

        weights = Vgg19().optmize_mask_and_weights(input_image=input_image, gussian_blur=gussian_blur)
        W_weights = Variable(weights, requires_grad = True)
        if torch.cuda.is_available():
            W_weights = W_weights.cuda()
        learining_rate = 0.01

        p_mask = torch.nn.Softmax(dim=0)
        output_probability = p_mask(self.model(input_image)[0]).detach()
        target_class = np.argmax(output_probability)


        for i in range(70):
            optimizer = torch.optim.Adam([W_weights], lr= learining_rate)
            optimizer.zero_grad()

            m_mask = self.create_m_mask(W_weights, input_image)
            background_mask = 1 - m_mask

            new_image_rep = (input_image * m_mask) + (gussian_blur * background_mask)
            new_image_rep_output = self.model(new_image_rep)[0, target_class]



            rep_background = (input_image * background_mask) + (gussian_blur * m_mask)
            background_rep_output = self.model(rep_background)[0,target_class]
            #output_background = p_mask(background_rep_output)[target_class]



            # Sum all to optimize
            loss = background_rep_output - new_image_rep_output + W_weights.sum()
            print(i, "second alg loss", loss)
            # Step
            loss.backward(retain_graph=True)

            optimizer.step()
            W_weights = Variable(torch.clamp(W_weights, min= 0.0), requires_grad=True)
            if torch.cuda.is_available():
                W_weights = W_weights.cuda()


            if i>0 and i % 10 == 0:
                learining_rate *= 1/2

        return  W_weights



if __name__ == '__main__':




    df = pd.read_csv("../fall11_urls.txt", delimiter="\t", encoding="latin1", header=None, names=["urls"]).reset_index()
    tar = tarfile.open("../Annotation.tar.gz")
    tarname = [tarinfo.name for tarinfo in tar]
    iou_list = []
    alpha_list = np.arange(0, 5.5, .5)
    for alpha in alpha_list:
        iou_inner_list = []
        for i in range(50, 70):
            tarname = tarname[i]
            annot = tar.extract(member=tarname)
            annot_files = tarfile.open(tarname)
            xml = [tarinfo.name for tarinfo in annot_files]
            for j in range(2, 7):
                try:
                    xml = xml[j]
                    xml_anot = annot_files.extract(member=xml)

                    annotation_dict={}

                    tree = ET.parse(xml)
                    root = tree.getroot()

                    annotation_dict.update({"name": root[1].text})
                    annotation_dict.update({"size":[i.text for i in root[3]]})
                    annotation_dict.update({"Bbox":[i.text for i in root[5][4]]})
                    annotation_dict.update({"url": df.loc[df["index"]== root[1].text, "urls"].values[0]})


                    print(annotation_dict)
                    #response = requests.get(annotation_dict["url"])
                    #img_pil = Image.open(io.BytesIO(response.content))
                    #img_pil = imageio.imread(io.BytesIO(response.content))
                    img = Image.open(requests.get(annotation_dict["url"], stream=True).raw)

                    resp = urllib.request.urlopen(annotation_dict["url"])
                    image = np.asarray(bytearray(resp.read()), dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)


                    blur = cv2.resize(image, (224, 224))
                    gussian_blur = find_gussian_blur(blur)


                    preprocessed_img = preprocess_image_eval(img)
                    final_mask = Disciminative_vgg19()

                    weights = final_mask.discriminative_mask_and_weights(preprocessed_img, gussian_blur)
                    mask = final_mask.create_m_mask(weights, preprocessed_img)
                    mask = torch.squeeze(mask, 0).permute(2, 1, 0)


                    image_size = annotation_dict["size"][0:2]
                    image_size = tuple([int(i) for i in image_size])


                    mask_tresh = copy.copy(mask.data.numpy())
                    mask_tresh = cv2.resize(mask_tresh, image_size)

                    mask_tresh = cv2.cvtColor(mask_tresh, cv2.COLOR_RGB2GRAY)
                    mask_tresh = cv2.convertScaleAbs(mask_tresh)
                    mean_intensity = mask_tresh.mean()
                    print(mean_intensity)

                    treshhold = alpha * mean_intensity

                    ret, thresh = cv2.threshold(mask_tresh, treshhold, 255, cv2.THRESH_BINARY)

                    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    print(len(contours))
                    c = max(contours, key=cv2.contourArea)
                    index = contours.index(c)
                    print(index)
                    cv2.drawContours(image, contours, -1, (100, 100, 255), 3)
                    x, y, w, h = cv2.boundingRect(c)

                    w = x + w
                    h = y + h
                    x_anot = int(annotation_dict["Bbox"][0])
                    y_anot = int(annotation_dict["Bbox"][1])
                    w_anot = int(annotation_dict["Bbox"][2])
                    h_anot = int(annotation_dict["Bbox"][3])

                    BOUNDING_BOX_pred = [x, y, w, h]
                    BOUNDING_BOX_ANOt = [x_anot, y_anot, w_anot, h_anot]
                    iou = bb_intersection_over_union(BOUNDING_BOX_ANOt, BOUNDING_BOX_pred)

                    iou_inner_list.append(iou)

                    cv2.rectangle(image, (x, y), (w, h), (100, 100, 255), 2)
                    cv2.rectangle(image, (x_anot, y_anot), (w_anot, h_anot), (0, 0, 255), 2)

                    #plt.imshow(image)
                    #plt.imshow(mask.detach(), "gray")
                    #plt.show()
                except:
                    pass
        print(iou_inner_list)
        iou_list.append(iou_inner_list)

    print(iou_list)



