import os

import numpy as np

from Code.Constants import FILES_DIR, HOC_MATRIX_FILE, HOG_MATRIX_FILE, VGG_BLOCK1_MATRIX_FILE, VGG_BLOCK2_MATRIX_FILE, \
    VGG_BLOCK3_MATRIX_FILE, STD_COLUMN, NPZ_EXTENSION
from sklearn.preprocessing import normalize
from skimage import img_as_ubyte
from skimage.transform import resize
from skimage import color

from Code.DataManager import DataManager as dm
import Code.FeatureExtractor as fe


class ImageProcessor:

    def __init__(self, update=False):
        """ Pre-processes the images and extracts several features: color,
         gradients and features from 3 layers of vgg16. Stores those features
         in files for later use. """

        if update:
            if not os.path.exists(FILES_DIR):
                os.makedirs(FILES_DIR)

            image_names = np.load(FILES_DIR + "image_names.npz")["names"]

            # Extracting HoC
            self.extract_feature(image_names, HOC_MATRIX_FILE, self.extract_img_hoc)

            # Extracting HoG
            self.extract_feature(image_names, HOG_MATRIX_FILE, self.extract_img_hog)

            # Extracting VGG16_block1
            self.extract_vgg_feature(image_names, VGG_BLOCK1_MATRIX_FILE, "block1_pool")

            # Extracting VGG16_block2
            self.extract_vgg_feature(image_names, VGG_BLOCK2_MATRIX_FILE, "block2_pool")

            # Extracting VGG16_block3
            self.extract_vgg_feature(image_names, VGG_BLOCK3_MATRIX_FILE, "block3_pool")

    def extract_img_hoc(self, img_name):
        img = dm.get_single_img(img_name)
        img = center_crop_image(img, size=224)
        img_hsv = color.rgb2hsv(img)
        img_int = img_as_ubyte(img_hsv)

        color_hist, bins = fe.hoc(img_int, bins=(4, 4, 4))
        color_feat = np.squeeze(normalize(color_hist.reshape(1, -1), norm="l2"))
        return color_feat

    def extract_img_hog(self, img_name):
        img = dm.get_single_img(img_name)
        img = center_crop_image(img, size=224)
        img_gray = color.rgb2gray(img)

        grad_hist = fe.my_hog(img_gray, orientations=8, pixels_per_cell=(32, 32))
        grad_feat = np.squeeze(normalize(grad_hist.reshape(1, -1), norm="l2"))
        return grad_feat

    def extract_img_vggblock(self, img_name, layer_name):
        img = dm.get_single_img(img_name)
        img = center_crop_image(img, size=224)

        vgg16 = fe.vgg16_layer(img, layer=layer_name)
        return vgg16

    def extract_feature(self, img_names, npz_name, function):
        features = []
        for img_name in img_names:
            features.append(function(img_name))

        features = np.array(features)
        np.savez('{}.npz'.format(FILES_DIR + npz_name), features=features)

    def extract_vgg_feature(self, img_names, npz_name, layer_name):
        features = []
        for img_name in img_names:
            features.append(self.extract_img_vggblock(img_name, layer_name))

        features = np.array(features)
        np.savez('{}.npz'.format(FILES_DIR + npz_name), features=features)


def load_feature(npz_name):
    # Reading feature matrices from files
    return np.load(FILES_DIR + npz_name + NPZ_EXTENSION, mmap_mode="r")[STD_COLUMN]


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

