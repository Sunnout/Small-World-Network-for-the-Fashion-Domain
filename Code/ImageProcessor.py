import os

import numpy as np
from skimage import color
from skimage import img_as_ubyte
from skimage.transform import resize
from sklearn.preprocessing import normalize

import Code.FeatureExtractor as fe
from Code.DataManager import DataManager as dm

from Code.Constants import FILES_DIR, HOC_MATRIX_FILE, HOG_MATRIX_FILE, FEATURE, NPZ_EXTENSION, VGG_MATRIX_FILES, \
    IMG_NAMES


class ImageProcessor:

    def __init__(self, update=False, layers=[]):
        """ Pre-processes the images and extracts several features: color,
         gradients and features from several VGG16 layers. Stores those features
         in npz files for later use. """

        if update:
            if not os.path.exists(FILES_DIR):
                os.makedirs(FILES_DIR)

            # Loading image names from file
            image_names = np.load(FILES_DIR + "image_names.npz")[IMG_NAMES]

            # Extracting and storing HoC
            self.extract_feature(image_names, HOC_MATRIX_FILE, self.extract_img_hoc)

            # Extracting and storing HoG
            self.extract_feature(image_names, HOG_MATRIX_FILE, self.extract_img_hog)

            # Extracting and storing VGG16 layers
            for layer in layers:
                self.extract_vgg_feature(image_names, VGG_MATRIX_FILES[layer], layer)

    @staticmethod
    def extract_feature(img_names, npz_name, function):
        """ Given the image names (img_names) of all the images in the database,
         extracts the features using a given function (function) for each image.
         This function can either extract the HoC or the HoG. Then, stores the
         features in an npz file with a given name (npz_name). """

        features = []
        for img_name in img_names:
            features.append(function(img_name))

        features = np.array(features)
        np.savez('{}.npz'.format(FILES_DIR + npz_name), features=features)

    @staticmethod
    def extract_img_hoc(img_name):
        """ Given an image name (img_name), fetches the image, pre-processes it
         and extracts its Histogram of Colors. Returns this feature. """

        # Fetching and pre-processing
        img = dm.get_single_img(img_name)
        img = center_crop_image(img, size=224)
        img_hsv = color.rgb2hsv(img)
        img_int = img_as_ubyte(img_hsv)

        # Extracting feature
        color_hist, bins = fe.hoc(img_int, bins=(4, 4, 4))
        color_feat = np.squeeze(normalize(color_hist.reshape(1, -1), norm="l2"))
        return color_feat

    @staticmethod
    def extract_img_hog(img_name):
        """ Given an image name (img_name), fetches the image, pre-processes it
         and extracts its Histogram of Oriented Gradients. Returns this feature. """

        # Fetching and pre-processing
        img = dm.get_single_img(img_name)
        img = center_crop_image(img, size=224)
        img_gray = color.rgb2gray(img)

        # Extracting feature
        grad_hist = fe.my_hog(img_gray, orientations=8, pixels_per_cell=(32, 32))
        grad_feat = np.squeeze(normalize(grad_hist.reshape(1, -1), norm="l2"))
        return grad_feat

    @staticmethod
    def extract_vgg_feature(img_names, npz_name, layer_name):
        """ Given the image names (img_names) of all the images in the database,
         extracts the features from a given VGG16 layer (layer_name) for each image.
         Then, stores the features in an npz file with a given name (npz_name). """

        features = []
        for img_name in img_names:
            img = dm.get_single_img(img_name)
            img = center_crop_image(img, size=224)

            features.append(fe.vgg16_layer(img, layer=layer_name))

        features = np.array(features)
        np.savez('{}.npz'.format(FILES_DIR + npz_name), features=features)


def load_feature(npz_name):
    """ Reads the feature matrix from a file with a given name (npz_name). """
    
    return np.load(FILES_DIR + npz_name + NPZ_EXTENSION, mmap_mode="r")[FEATURE]


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
