import os
import random

import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize

from keras.preprocessing import image


def get_img_names():
    img_list = os.listdir('Data/')

    return img_list


def get_img(img_name):
    return imread('Data/'+img_name)


def get_rand_image():
    img_list = get_img_names()
    return random.choice(img_list)


def center_crop_image(im, size=224):

    if im.shape[2] == 4:  # Remove the alpha channel
        im = im[:, :, 0:3]

    # Resize so smallest dim = 224, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = resize(image=im, output_shape=(224, int(w * 224 / h)))
    else:
        im = resize(im, (int(h * 224 / w), 224))

    # Center crop to 224x224
    h, w, _ = im.shape
    im = im[h // 2 - 112:h // 2 + 112, w // 2 - 112:w // 2 + 112]

    return im
