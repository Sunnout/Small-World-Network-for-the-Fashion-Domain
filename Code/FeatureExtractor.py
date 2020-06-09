import numpy as np
from skimage.feature import hog

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf


def hoc(im, bins=(4, 4, 4), hist_range=(256, 256, 256)):
    """ Extracts the histogram of colours of a given image (img). """

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


def my_hog(img, orientations=8, pixels_per_cell=(16, 16)):
    """ Extracts histogram of oriented gradients of a given image (img). """

    fd, hog_image = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell, visualize=True)
    return fd


def vgg16_layer(img, layer='block2_pool'):
    """ Extracts the feature values for a given image (img), from a given VGG16
      layer (layer). """

    tf.keras.backend.clear_session()
    print("VGG16 is processing a new image!")
    
    # Extracting features from the VGG16 layer
    model = VGG16(weights='imagenet', include_top=True)
    model_layer = Model(inputs=model.input, outputs=model.get_layer(layer).output)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model_layer.predict(x)

    # Max Pooling the extracted features to reduce their size
    max_pool_3d = tf.keras.layers.MaxPooling3D(pool_size=(8, 8, 8), strides=(1, 1, 1), padding='valid')
    pool_feat = tf.constant(features)
    pool_feat = tf.reshape(pool_feat, [1, pool_feat.shape[1], pool_feat.shape[2], pool_feat.shape[3], 1])
    pool_feat = max_pool_3d(pool_feat)
    pool_feat = tf.reshape(pool_feat, [-1])
    pool_feat = pool_feat.numpy()

    return pool_feat.flatten()
