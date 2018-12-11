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


image_dataset = []
df = pd.read_csv("../fall11_urls.txt", delimiter="\t", encoding="latin1", header=None, names=["urls"]).reset_index()
tar = tarfile.open("../Annotation.tar.gz")
tarname_list = [tarinfo.name for tarinfo in tar]

for tarname in tarname_list[50: 60]:
    annot = tar.extract(member=tarname)
    annot_files = tarfile.open(tarname)
    xml_list = [tarinfo.name for tarinfo in annot_files]
    for xml in xml_list[2:12]:
        annotation_dict = {}
        try:
            xml_anot = annot_files.extract(member=xml)
            tree = ET.parse(xml)
            root = tree.getroot()

            annotation_dict.update({"name": root[1].text})
            annotation_dict.update({"size": [i.text for i in root[3]]})
            annotation_dict.update({"Bbox": [i.text for i in root[5][4]]})
            annotation_dict.update({"url": df.loc[df["index"] == root[1].text, "urls"].values[0]})

            image_dataset.append(annotation_dict)
            print(tarname_list.index(tarname), xml_list.index(xml))
        except:
            pass


print(image_dataset)
df = pd.DataFrame.from_dict(image_dataset)
df.to_csv("picture_data.csv")