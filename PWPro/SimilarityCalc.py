import matplotlib.pyplot as plt
import numpy as np

from skimage.feature import hog
from skimage import data, exposure
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize

from keras.preprocessing import image

# Straight forward HoC implementation on RGB space
# For a more complete implementation, with better parametrization, etc., you can check the OpenCV library.
def hoc(im, bins=(16, 16, 16), hist_range=(256, 256, 256)):
    im_r = im[:, :, 0]
    im_g = im[:, :, 1]
    im_b = im[:, :, 2]

    red_level = hist_range[0] / bins[0]
    green_level = hist_range[1] / bins[1]
    blue_level = hist_range[2] / bins[2]

    im_red_levels = np.floor(im_r / red_level)
    im_green_levels = np.floor(im_g / green_level)
    im_blue_levels = np.floor(im_b / blue_level)

    ind = im_blue_levels * bins[0] * bins[1] + im_green_levels * bins[0] + im_red_levels

    hist_r, bins_r = np.histogram(ind.flatten(), bins[0] * bins[1] * bins[2])

    return hist_r, bins_r

def calc_sim_matrix(feature='hoc'):
