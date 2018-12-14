import os
import copy

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

    def discriminative_mask_and_weights(self, input_image, gussian_blur, target_class):

        weights = Vgg19().optmize_mask_and_weights(input_image=input_image, gussian_blur=gussian_blur)
        W_weights = Variable(weights, requires_grad = True)
        if torch.cuda.is_available():
            W_weights = W_weights.cuda()
        learining_rate = 0.01

        p_mask = torch.nn.Softmax(dim=0)
        output_probability = p_mask(self.model(input_image)[0]).detach()
        print("output probability",np.argmax(output_probability), output_probability[np.argmax(output_probability)])

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
    # Get params
    original_image_dir = "parot.jpg"

    directory = "input_images/" + original_image_dir
    original_image = preprocess_image(directory)
    target_class = 243

    image = cv2.imread(directory)
    blur = cv2.resize(image, (224, 224))
    gussian_blur = find_gussian_blur(blur)


    final_mask = Disciminative_vgg19()
     # width & height
    weights = final_mask.discriminative_mask_and_weights(original_image, gussian_blur, target_class)
    mask = final_mask.create_m_mask(weights, original_image)
    #new_image_rep = (original_image * mask) + (gussian_blur * (1-mask))

    gussian_blur = torch.squeeze(gussian_blur, 0).permute(2,1,0)

    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1 / 0.229, 1 / 0.224, 1 / 0.225]

    recreated_im = copy.copy(mask.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    #recreated_im[recreated_im > 0] = 1
    #recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im  * 400)
    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)


    mask_heatmap = copy.copy(mask.data.numpy()[0])
    color_map = mpl_color_map.get_cmap('hsv_r')
    heatmap = color_map(mask_heatmap[0, :, :])
    heatmap[:, :, 3] = 0.5
    heatmap = Image.fromarray((heatmap * 255).astype("uint8"))
    heatmap.save("generated/heatmap_discriminative" + original_image_dir[:-4] + ".png")




    cv2.imwrite('generated/Inv_Image_Layer_2nd' + original_image_dir, recreated_im)
    cv2.imwrite("generated/inv_image_2nd" + original_image_dir, recreated_im[:, :, 2])
    cv2.imwrite("generated/background_mask_2nd" + original_image_dir, recreate_image(1 - mask))




