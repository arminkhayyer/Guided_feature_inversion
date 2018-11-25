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

def preprocess_image(image):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image

    transform = transforms.Compose([
        #transforms.Scale(size=256),
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    img = transform(Image.open(image)).unsqueeze(0)
    #img = torch.from_numpy(img).float()

    return img


def find_gussian_blur(image):
    blur = cv2.GaussianBlur(image, ksize=(23,23) ,sigmaX=11)
    blur = np.float32(blur)
    blur = torch.from_numpy(blur)
    blur = (torch.unsqueeze(blur.permute(2, 1, 0), 0))
    return blur

def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    #recreated_im[recreated_im > 1] = 1
    #recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 400)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    # Convert RBG to GBR
    #recreated_im = recreated_im[..., ::-1]
    return recreated_im


'''
img= "snake.jpg"
img = preprocess_image(img)
model= models.vgg19(pretrained=True)

def forward(x):
    """Extract multiple convolutional feature maps."""
    features = []
    for name, layer in enumerate(model.features):
        x = layer(x)
        if name in [35, 36]:
            features.append(x)
    return features

print((forward(img)[1].size()))
print((forward(img)[0].size()))


base_layer = forward(img)[0]
W_weights = Variable(.1 * torch.ones(512), requires_grad= True)
#print(W_weights)

m_mask = torch.zeros(14, 14)
#print(m_mask)
for i, key in enumerate(base_layer):
    for j, b in enumerate(key):
        m_mask += W_weights[j] * b
m_mask = m_mask.view(1, 1, 14, 14)
upsample = nn.Upsample(size=(224, 224), mode="bilinear")
m_mask=upsample(m_mask)
m_mask=torch.cat((m_mask , m_mask, m_mask), 1)

print(m_mask.size())'''









class Vgg19():


    def __init__(self):
        super(Vgg19, self).__init__()
        self.model= models.vgg19(pretrained=True)

        self.model.eval()
        self.select_layers= {"feature_inversion":36, "base_layer":35}
        if not os.path.exists('generated'):
            os.makedirs('generated')



    def euclidian_loss(self, org_matrix, target_matrix):
        """
            Euclidian loss is the main loss function in the paper
            ||fi(x) - fi(x_0)||_2^2
        """
        distance_matrix = target_matrix - org_matrix
        euclidian_distance = ((distance_matrix.view(-1))**2).sum()
        return euclidian_distance


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

    def optmize_mask_and_weights(self, input_image, gussian_blur):

        W_weights = Variable(.1 * torch.ones(512), requires_grad = True)
        base_layer = self.forward(input_image)[1]

        #optimizer = torch.optim.Adam([W_weights], lr=.01)
        # Define optimizer for previously created image
        #optimizer = torch.optim.SGD([W_weights], lr=.01)
        # Get the output from the model after a forward pass until target_layer
        # with the input image (real image, NOT the randomly generated one)
        input_image_layer_output_l0 = self.forward(input_image)[1]

        for i in range(10):
            optimizer = torch.optim.Adam([W_weights], lr=.01)
            optimizer.zero_grad()
            # Get the output from the model after a forward pass until target_layer
            # with the generated image (randomly generated one, NOT the real image)

            # defining the m mask and weights
            m_mask = self.create_m_mask(W_weights, input_image)

            background_mask = 1 - m_mask

            new_image_rep = (input_image * m_mask) + (gussian_blur * background_mask)
            new_image_rep_output = self.forward(new_image_rep)[1]


            # Calculate euclidian loss
            euc_loss = self.euclidian_loss(input_image_layer_output_l0, new_image_rep_output)
            # Calculate alpha regularization

            # Sum all to optimize
            loss = euc_loss  + 10 * (W_weights).sum()
            print(i, "first algorithm loss", loss)
            # Step
            loss.backward(retain_graph=True)

            optimizer.step()
            W_weights = Variable(torch.clamp(W_weights, min= 0.0), requires_grad=True)
            print(i,"first algorithm weights sum after clipping", W_weights.sum())


        return  W_weights



if __name__ == '__main__':
    # Get params
    original_image_dir = "cat_dog.png"
    directory = "input_images/" + original_image_dir
    original_image = preprocess_image(directory)
    gussian_blur = find_gussian_blur(cv2.imread(directory))

    final_mask = Vgg19()
     # width & height
    weights = final_mask.optmize_mask_and_weights(original_image, gussian_blur)
    mask = final_mask.create_m_mask(weights, original_image)
    new_image_rep = (original_image * mask) + (gussian_blur * (1-mask))

    gussian_blur = torch.squeeze(gussian_blur, 0).permute(2,1,0)

    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1 / 0.229, 1 / 0.224, 1 / 0.225]

    recreated_im = copy.copy(mask.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    #recreated_im[recreated_im > .5] = 1
    #recreated_im[recreated_im < .5] = 0
    recreated_im = np.round(recreated_im  * 400)
    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)

    mask_heatmap= copy.copy(mask.data.numpy()[0])
    #for c in range(3):
     #   mask_heatmap[c] /= reverse_std[c]
    #    mask_heatmap[c] -= reverse_mean[c]

    color_map = mpl_color_map.get_cmap('hsv_r')
    heatmap = color_map(mask_heatmap[0, :,:])
    heatmap[:, :, 3] = 0.5
    heatmap = Image.fromarray((heatmap * 255).astype("uint8"))
    heatmap.save("generated/heatmap_"+ original_image_dir[:-4]+".png")



    cv2.imwrite('generated/Inv_Image_Layer_' + original_image_dir, recreated_im)

    cv2.imwrite("generated/inv_image_"+ original_image_dir, recreated_im[:, :, 2])
    cv2.imwrite("generated/background_mask_" + original_image_dir, recreate_image(1-mask))




