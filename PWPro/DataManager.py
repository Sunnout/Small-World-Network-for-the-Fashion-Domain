import os
import random

from skimage.io import imread


def get_img_names():
    img_list = []
    for entry in os.scandir("../Data"):
        img_list.append(entry.name);

    img_list.sort()
    return img_list


def get_img(img_name):
    return imread("../Data/"+img_name)


def get_rand_image():
    img_list = get_img_names()
    img_name = random.choice(img_list)
    return imread("../Data/"+img_name)
