import os

import numpy as np
from sklearn.preprocessing import normalize
from skimage import img_as_ubyte
from skimage.transform import resize
from skimage import color

from DataManager import DataManager as dm
import FeatureExtractor as fe


# Directory where we save the output files
FILES_DIR = "../Files/"


class ImageProcessor:

    def __init__(self, update=False):
        """ Pre-processes the images and extracts several features: color,
         gradients and features from 3 layers of vgg16. Stores those features
         in files for later use. """

        self.image_names = np.load(FILES_DIR + "image_names.npz")["names"]

        if update:
            for img_name in self.image_names:
                img = dm.get_single_img(img_name)

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

                # Extracting VGG16_block1
                vgg16 = fe.vgg16_layer(img, layer= "block1_pool")
                self.vgg_block1.append(vgg16)

                # Extracting VGG16_block2
                vgg16 = fe.vgg16_layer(img, layer='block2_pool')
                self.vgg_block2.append(vgg16)

                # Extracting VGG16_block3
                vgg16 = fe.vgg16_layer(img, layer='block3_pool')
                self.vgg_block3.append(vgg16)

            # Saving feature matrices in files
            self.colors = np.array(self.colors)
            self.grads = np.array(self.grads)
            self.vgg_block1 = np.array(self.vgg_block1)
            self.vgg_block2 = np.array(self.vgg_block2)
            self.vgg_block3 = np.array(self.vgg_block3)

            if not os.path.exists(FILES_DIR):
                os.makedirs(FILES_DIR)

            np.savez('{}.npz'.format(FILES_DIR + "hoc_matrix"), hoc=self.colors)
            np.savez('{}.npz'.format(FILES_DIR + "hog_matrix"), hog=self.grads)
            np.savez('{}.npz'.format(FILES_DIR + "vgg16_block1_matrix"), b1=self.vgg_block1)
            np.savez('{}.npz'.format(FILES_DIR + "vgg16_block2_matrix"), b2=self.vgg_block2)
            np.savez('{}.npz'.format(FILES_DIR + "vgg16_block3_matrix"), b3=self.vgg_block3)

        else:
            # Reading feature matrices from files
            self.colors = np.load(FILES_DIR + "hoc_matrix.npz")["hoc"]
            self.grads = np.load(FILES_DIR + "hog_matrix.npz")["hog"]
            self.vgg_block1 = np.load(FILES_DIR + "vgg16_block1_matrix.npz")["b1"]
            self.vgg_block2 = np.load(FILES_DIR + "vgg16_block2_matrix.npz")["b2"] # Mudar para b2 quando correr com True outra vez
            self.vgg_block3 = np.load(FILES_DIR + "vgg16_block3_matrix.npz")["b3"]

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
