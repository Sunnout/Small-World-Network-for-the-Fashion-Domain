import os
import random

from skimage.io import imread


def get_img_names():
    path = os.getcwd()+"/../Data"
    os.chdir(path)
    img_list = os.listdir()
    print(img_list)

    return img_list


def get_img(img_name):
    return imread(img_name)


def get_rand_image():
    img_list = get_img_names()
    img_name = random.choice(img_list)
    return imread(img_name)



