import matplotlib.pyplot as plt
import numpy as np

from skimage.feature import hog
from skimage import exposure


# Straight forward HoC implementation on RGB space
# For a more complete implementation, with better parametrization, etc., you can check the OpenCV library.
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
    
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.bar(bins_r[:-1], hist_r ,width=1)
    ax.set_xticks([])
    ax.set_xlim(bins_r[:-1].min()*-2, max(bins_r.max(), hist_r.shape[0]*1.3))
    #ax.set_ylim(0, 2000)
    plt.show()

    return hist_r, bins_r


def my_hog(img, orientations=8, pixels_per_cell=(16, 16)):
    #convert gray scale??
    
    fd, hog_image = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell, visualize=True)
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

    return fd
