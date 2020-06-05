import os
import random

import numpy as np
from skimage.io import imread

from Code.Constants import FILES_DIR, DATA_DIR, IMG_NAMES


class DataManager:

    def __init__(self, update=False):
        """ Updates or reads the names of the images of the database. """

        self.image_names = []

        # Updates image names if update=True
        if update:
            self.image_names = self.get_img_names()

            if not os.path.exists(FILES_DIR):
                os.makedirs(FILES_DIR)

            np.savez('{}.npz'.format(FILES_DIR + "image_names"), names=self.image_names)
        else:
            self.image_names = np.load(FILES_DIR + "image_names.npz")[IMG_NAMES]

    def get_all_imgs(self):
        """ Returns all the images from the database. """

        img_list = []
        for name in self.image_names:
            img_list.append(imread(DATA_DIR + name))

        return img_list

    def get_rand_image(self):
        """ Returns a random image from the database. """

        img_list = self.get_all_imgs()
        return random.choice(img_list)

    def get_num_imgs(self):
        """ Returns the number of images in the database. """

        return len(self.image_names)

    @staticmethod
    def get_img_names():
        """ Returns the list of sorted image names. """

        name_list = []
        for entry in os.scandir(DATA_DIR):
            name_list.append(entry.name)

        name_list.sort()
        return name_list

    @staticmethod
    def get_img_index(img_name):
        """ Prints and returns a tuple of (index, image_name), given an image
         name (img_name). """

        idx = DataManager.get_img_names().index(img_name)
        print("(" + str(idx) + ", " + img_name + ")")
        return (idx, img_name)

    @staticmethod
    def get_single_img(img_name):
        """ Reads and returns an image, given its name (img_name). """

        return imread(DATA_DIR + img_name)

    @staticmethod
    def get_img_path(img_name):
        """ Returns the full path of an image, given its name (img_name). """

        return DATA_DIR + img_name

    def get_rand_set(self, size):
        """ Returns a set of random image indexes of a given size (size). """

        num_imgs = self.get_num_imgs()
        if size > num_imgs:
            return range(0, num_imgs)
        return random.sample(range(0, num_imgs), size)
