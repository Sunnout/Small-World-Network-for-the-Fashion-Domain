import os
import random
import numpy as np
from skimage.io import imread

DIR = "../Data/"


class DataManager:

    def __init__(self, update=False):
        self.image_names = []
        # Updates image names if update=True
        if update:
            self.image_names = self.get_img_names()
            print(self.image_names)
            np.savez('{}.npz'.format("image_names"), names=self.image_names)
        else:
            self.image_names = np.load("image_names.npz")["names"]

    def initialize(self, update=False):
        # Updates image names if update=True
        if update:
            self.image_names = self.get_img_names()
            print(self.image_names)
            np.savez('{}.npz'.format("image_names"), names=self.image_names)
        else:
            self.image_names = np.load("image_names.npz")["names"]

    def get_all_imgs(self):
        img_list = []
        for name in self.image_names:
            img_list.append(imread(DIR + name))

        return img_list

    def get_rand_image(self):
        img_list = self.get_all_imgs()
        return random.choice(img_list)

    def get_num_imgs(self):
        return len(self.image_names)

    @staticmethod
    def get_img_names():
        name_list = []
        for entry in os.scandir(DIR):
            name_list.append(entry.name)

        name_list.sort()
        return name_list

    @staticmethod
    def get_img_index(img):
        print("("+str(DataManager.get_img_names().index(img))+", "+img+")")
        return (DataManager.get_img_names().index(img), img)


    @staticmethod
    def get_single_img(img_name):
        return imread(DIR + img_name)

    @staticmethod
    def get_img_path(img_name):
        return DIR + img_name

    def get_rand_set(self, size):
        num_imgs = self.get_num_imgs()
        if size > num_imgs:
            return range(0, num_imgs)
        return random.sample(range(0, num_imgs), size)
