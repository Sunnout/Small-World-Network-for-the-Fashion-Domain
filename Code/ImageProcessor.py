import os

import numpy as np
from sklearn.preprocessing import normalize
from skimage import img_as_ubyte
from skimage.transform import resize
from skimage import color

from Code.DataManager import DataManager as dm
import Code.FeatureExtractor as fe


# Directory where we save the output files
FILES_DIR = "Files/"


class ImageProcessor:

    def __init__(self, update=False):
        """ Pre-processes the images and extracts several features: color,
         gradients and features from 3 layers of vgg16. Stores those features
         in files for later use. """

        self.colors = []
        self.grads = []
        self.vgg_block1 = []
        self.vgg_block2 = []
        self.vgg_block3 = []


        image_names = np.load(FILES_DIR + "image_names.npz")["names"]

        if update:
            if not os.path.exists(FILES_DIR):
                os.makedirs(FILES_DIR)

            # Extracting HoC
            self.extract_feature(image_names, "hoc_matrix", self.extract_img_hoc)

            # Extracting HoG
            self.extract_feature(image_names, "hog_matrix", self.extract_img_hog)

            # Extracting VGG16_block1
            self.extract_vgg_feature(image_names, "vgg16_block1_matrix", "block1_pool")

            # Extracting VGG16_block2
            self.extract_vgg_feature(image_names, "vgg16_block2_matrix", "block2_pool")

            # Extracting VGG16_block3
            self.extract_vgg_feature(image_names, "vgg16_block3_matrix", "block3_pool")

        else:
            # Reading feature matrices from files
            self.colors = np.load(FILES_DIR + "hoc_matrix.npz")["hoc"]
            self.grads = np.load(FILES_DIR + "hog_matrix.npz")["hog"]
            self.vgg_block1 = np.load(FILES_DIR + "vgg16_block1_matrix.npz")["b1"]
            self.vgg_block2 = np.load(FILES_DIR + "vgg16_block2_matrix.npz")["b2"] # Mudar para b2 quando correr com True outra vez
            self.vgg_block3 = np.load(FILES_DIR + "vgg16_block3_matrix.npz")["b3"]

    def extract_img_hoc(self, img_name):
        img = dm.get_single_img(img_name)
        img = self.center_crop_image(img, size=224)
        img_hsv = color.rgb2hsv(img)
        img_int = img_as_ubyte(img_hsv)

        color_hist, bins = fe.hoc(img_int, bins=(4, 4, 4))
        color_feat = np.squeeze(normalize(color_hist.reshape(1, -1), norm="l2"))
        return color_feat

    def extract_img_hog(self, img_name):
        img = dm.get_single_img(img_name)
        img = self.center_crop_image(img, size=224)
        img_gray = color.rgb2gray(img)

        grad_hist = fe.my_hog(img_gray, orientations=8, pixels_per_cell=(32, 32))
        grad_feat = np.squeeze(normalize(grad_hist.reshape(1, -1), norm="l2"))
        return grad_feat

    def extract_img_vggblock(self, img_name, layer_name):
        img = dm.get_single_img(img_name)
        img = self.center_crop_image(img, size=224)

        vgg16 = fe.vgg16_layer(img, layer=layer_name)
        return vgg16

    def extract_feature(self, img_names, npz_name, function):
        features = []
        for img_name in img_names:
            features.append(function(img_name))

        features = np.array(features)
        np.savez('{}.npz'.format(FILES_DIR + npz_name), feature=features)

    def extract_vgg_feature(self, img_names, npz_name, layer_name):
        features = []
        for img_name in img_names:
            features.append(self.extract_img_vggblock(img_name, layer_name))

        features = np.array(features)
        np.savez('{}.npz'.format(FILES_DIR + npz_name), feature=features)

    @staticmethod
    def load_feature(npz_name):
        return np.load(FILES_DIR + npz_name + ".npz")["features"]

    @staticmethod
    def center_crop_image(im, size=224):
        """ Removes the alpha channel from an images (img), centers it and crops
         it to a size of 224x224. Returns the processed image. """

        if im.shape[2] == 4:  # Remove the alpha channel
            im = im[:, :, 0:3]

        # Resize so smallest dim = 224, preserving aspect ratio
        h, w, _ = im.shape
        if h < w:
            im = resize(image=im, output_shape=(size, int(w * size / h)))
        else:
            im = resize(im, (int(h * size / w), size))

        # Center crop to 224x224
        h, w, _ = im.shape
        im = im[h // 2 - 112:h // 2 + 112, w // 2 - 112:w // 2 + 112]

        return im
