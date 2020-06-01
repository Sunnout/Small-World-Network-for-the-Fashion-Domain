import os

from skimage import img_as_ubyte
from skimage.transform import resize
import numpy as np
from skimage import color
from sklearn.preprocessing import normalize
from DataManager import DataManager as dm
import FeatureExtractor as fe


class ImageProcessor:

    def __init__(self, update=False):
        self.image_names = []
        self.colors = []
        self.grads = []
        self.vgg_block1 = []
        self.vgg_block2 = []

        self.image_names = np.load("image_names.npz")["names"]

        # Updates the feature matrices if update=True
        if update:

            for img_name in self.image_names:
                img = dm.get_single_img(img_name)
                print(img_name)
                # Extracting HoC
                img = self.center_crop_image(img, size=224)
                img_hsv = color.rgb2hsv(img)
                img_int = img_as_ubyte(img_hsv)
                color_hist, bins = fe.hoc(img_int, bins=(4, 4, 4))
                color_feat = np.squeeze(normalize(color_hist.reshape(1, -1), norm="l2"))
                self.colors.append(color_feat)

                # Extracting HoG
                img_gray = color.rgb2gray(img)
                grad_hist = fe.my_hog(img_gray, orientations=8, pixels_per_cell=(32, 32))
                grad_feat = np.squeeze(normalize(grad_hist.reshape(1, -1), norm="l2"))
                self.grads.append(grad_feat)

                # Extracting VGG16
                vgg16 = fe.vgg16_layer(img, layer= "block1_pool")
                self.vgg_block1.append(vgg16)

                vgg16 = fe.vgg16_layer(img, layer='block2_pool')
                self.vgg_block2.append(vgg16)

            # Saving feature matrices in files
            self.colors = np.array(self.colors)
            self.grads = np.array(self.grads)
            self.vgg_block1 = np.array(self.vgg_block1)
            self.vgg_block2 = np.array(self.vgg_block2)
            np.savez('{}.npz'.format("hoc_matrix"), hoc=self.colors)
            np.savez('{}.npz'.format("hog_matrix"), hog=self.grads)
            np.savez('{}.npz'.format("vgg16_block1_matrix"), b1=self.vgg_block1)
            np.savez('{}.npz'.format("vgg16_block2_matrix"), b1=self.vgg_block2)

        else:
            # Reading feature matrices from files
            self.colors = np.load("hoc_matrix.npz")["hoc"]
            self.grads = np.load("hog_matrix.npz")["hog"]
            self.vgg_block1 = np.load("vgg16_block1_matrix.npz")["b1"]
            self.vgg_block2 = np.load("vgg16_block2_matrix.npz")["b1"]

    def initialize(self, update=False):
        self.image_names = np.load("image_names.npz")["names"]

        # Updates the feature matrices if update=True
        if update:

            for img_name in self.image_names:
                img = dm.get_single_img(img_name)
                print(img_name)
                # Extracting HoC
                img = self.center_crop_image(img, size=224)
                img_hsv = color.rgb2hsv(img)
                img_int = img_as_ubyte(img_hsv)
                color_hist, bins = fe.hoc(img_int, bins=(4, 4, 4))
                color_feat = np.squeeze(normalize(color_hist.reshape(1, -1), norm="l2"))
                self.colors.append(color_feat)

                # Extracting HoG
                img_gray = color.rgb2gray(img)
                grad_hist = fe.my_hog(img_gray, orientations=8, pixels_per_cell=(32, 32))
                grad_feat = np.squeeze(normalize(grad_hist.reshape(1, -1), norm="l2"))
                self.grads.append(grad_feat)

                # Extracting VGG16
                vgg16 = fe.vgg16_layer(img)
                self.vgg_block1.append(vgg16)

            # Saving feature matrices in files
            self.colors = np.array(self.colors)
            self.grads = np.array(self.grads)
            self.vgg_block1 = np.array(self.vgg_block1)
            np.savez('{}.npz'.format("hoc_matrix"), hoc=self.colors)
            np.savez('{}.npz'.format("hog_matrix"), hog=self.grads)
            np.savez('{}.npz'.format("vgg16_block1_matrix"), b1=self.vgg_block1)

        else:
            # Reading feature matrices from files
            self.colors = np.load("hoc_matrix.npz")["hoc"]
            self.grads = np.load("hog_matrix.npz")["hog"]
            self.vgg_block1 = np.load("vgg16_block1_matrix.npz")["b1"]

    @staticmethod
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
