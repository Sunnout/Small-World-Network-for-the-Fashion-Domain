import numpy as np

from sklearn.metrics import pairwise_distances
from DataManager import DataManager
from ImageProcessor import ImageProcessor


class SimilarityCalculator:

    def __init__(self, kn=10):
        dm = DataManager(True)
        ip = ImageProcessor(True)

        color_matrix = ip.colors
        grads_matrix = ip.grads

        self.sim_matrix_colors = []
        self.sim_matrix_grads = []

        # for i in range(0, dm.get_num_imgs() - 1):
        for i in range(0, 1):
            indexes_colors, dists_colors = k_neighbours(q=color_matrix[i].reshape(1, -1), X=color_matrix,
                                                        metric="euclidean", k=kn)

            #GET IMAGE NAMES ESTA MAL!!
            self.sim_matrix_colors.append(list(zip(indexes_colors, dists_colors, dm.get_img_names()[i])))

            indexes_grads, dists_grads = k_neighbours(q=grads_matrix[i].reshape(1, -1), X=grads_matrix,
                                                      metric="euclidean", k=kn)

            self.sim_matrix_grads.append(list(zip(indexes_grads, dists_grads, dm.get_img_names()[i])))


def k_neighbours(q, X, metric="euclidean", k=10):
    dists = pairwise_distances(q, X, metric=metric)

    # Dists gets a shape 1 x NumDocs. Convert it to shape NumDocs (i.e. drop the first dimension)
    dists = np.squeeze(dists)
    sorted_indexes = np.argsort(dists)

    return sorted_indexes[:k], dists[sorted_indexes[:k]]


sc = SimilarityCalculator(kn=5)
print(sc.sim_matrix_colors)
