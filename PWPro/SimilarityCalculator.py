import numpy as np

from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
import DataManager as dm


def k_neighbours(q, X, metric="euclidean", k=10):
    # Check pairwise_distances function docs: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances
    dists = pairwise_distances(q, X, metric=metric)

    # Dists gets a shape 1 x NumDocs. Convert it to shape NumDocs (i.e. drop the first dimension)
    dists = np.squeeze(dists)
    sorted_indexes = np.argsort(dists)

    return sorted_indexes[:k], dists[sorted_indexes[:k]]


def calc_sim_matrix(feature='hoc'):
    images = dm.get_img_names()

    feats = []
    for img in images:
        img = dm.get_img(img)

        # resize image
        img = dm.center_crop_image(img, size=224)

        res = []
        if feature == 'hoc':
            """TODO: Apply Histogram of Colors"""
        else:
            if feature == 'hog':
                """TODO: Apply Histogram of Gradients"""
            else:
                """TODO: Create exit clause"""

        # Normalize features
        # We add 1 dimension to comply with scikit-learn API
        re = np.squeeze(normalize(res.reshape(1, -1), norm="l2"))

        feats.append(res)

    return np.array(feats)
