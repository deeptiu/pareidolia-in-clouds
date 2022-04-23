from __future__ import print_function
from matplotlib.image import imsave

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import os
import copy
import cv2

from neural_style_transfer import *

def train_one_img(style_filename, content_filename, save_filename, device, imsize):
    loader_style, loader_content = transforms_img(imsize)
    style_img = image_loader(style_filename, loader_style, device)
    content_img = image_loader(content_filename, loader_content, device)

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    input_img = content_img.clone()

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, device, num_steps=1500)

    output = output.squeeze()#.detach().cpu().numpy()
    save_image(output, save_filename)


if __name__ == "__main__":

    device = set_cuda_device(3)
    # desired size of the output image
    imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

    translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", 
                "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", 
                "mucca": "cow", "pecora": "sheep", "ragno": "spider", "scoiattolo": "squirrel", 
                "dog": "cane", "cavallo": "horse", "elephant" : "elefante", 
                "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto", 
                "cow": "mucca", "spider": "ragno", "squirrel": "scoiattolo"}

    data_path = "raw-img/"
    save_path = "generated_data/"
    style_filename = "./data/cloud.jpeg"

    for root, dirs, files in os.walk(data_path):
        animal = root[len(data_path):]
        
        if len(animal) > 1:
            translated_animal = translate[animal]
            for name in files:
                content_filename = data_path + animal + "/" + name
                save_filename = save_path + translated_animal + "/" + name
                train_one_img(style_filename, content_filename, save_filename, device, imsize)




