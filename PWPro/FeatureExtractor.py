import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

from skimage.feature import hog
from skimage import exposure

from DataManager import DataManager as dm


def hoc(im, bins=(4, 4, 4), hist_range=(256, 256, 256)):
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
    fd, hog_image = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell, visualize=True)
    return fd


def plot_hoc(hist_r, bins_r):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.bar(bins_r[:-1], hist_r, width=1)
    ax.set_xticks([])
    ax.set_xlim(bins_r[:-1].min() * -2, max(bins_r.max(), hist_r.shape[0] * 1.3))
    # ax.set_ylim(0, 2000)
    plt.show()


def plot_hog(fd, hog_image, img):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title('Input image', fontsize=20)

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients', fontsize=18)
    plt.show()


def vgg16_layer(img):
    model = VGG16(weights='imagenet', include_top=True)
    model_layer = Model(inputs=model.input, outputs=model.get_layer('fc1').output)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model_layer.predict(x)

    from keras.applications.vgg16 import decode_predictions
    # convert the probabilities to class labels
    #label = decode_predictions(features)
    # retrieve the most likely result, e.g. highest probability
    # label = label[0][0]
    # print the classification
    #print('%s (%.2f%%)' % (label[1], label[2] * 100))
    print(features)
