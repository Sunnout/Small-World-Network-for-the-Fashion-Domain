import os

from skimage import img_as_ubyte
from skimage.transform import resize
import numpy as np
from DataManager import DataManager as dm
import FeatureExtractor as fe


class ImageProcessor:

    def __init__(self, update=False):

        self.image_names = np.load("image_names.npz")["names"]

        # Only updates the feature matrices if we call ImageProcessor(update=True)
        if update:
            self.colors = []
            self.grads = []

            for img_name in self.image_names:
                img = dm.get_single_img(img_name)

                # Pre-processing Image
                img = self.center_crop_image(img, size=224)

                # Extract HoG
                grad_hist = fe.my_hog(img)  # gray scale?
                self.grads.append(grad_hist)
                np.savez('{}.npz'.format("hog_matrix"), hoc=self.grads)

                # Extract HoC
                img_int = img_as_ubyte(img)
                color_hist, bins = fe.hoc(img_int, bins=(4, 4, 4))
                self.colors.append(color_hist)
                np.savez('{}.npz'.format("hoc_matrix"), hoc=self.colors)
        else:
            self.colors = np.load("hoc_matrix.npz")["hoc"]
            self.grads = np.load("hog_matrix.npz")["hog"]

    #Acho que não vamos precisar disto
    def process_single_img(self, img_name, img):
        """Função para processar imagem de input
        Busca às features das imagens da bd se a imagem pertencer à bd
        Senão calcula as features para a imagem nova"""

        if img_name in self.image_names:
            # Get features from DB
            img_color = self.colors.index(img_name)
            img_grad = self.grads.index(img_name)
        else:
            # Extract features
            img = self.center_crop_image(img, size=224)
            img_color, bins = fe.hoc(img)
            img_grad = fe.my_hog(img)  # gray scale?

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