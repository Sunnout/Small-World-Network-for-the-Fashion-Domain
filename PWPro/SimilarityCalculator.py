import numpy as np

from sklearn.metrics import pairwise_distances
from DataManager import DataManager
from ImageProcessor import ImageProcessor


class SimilarityCalculator:

    def __init__(self, update=False, kn=10):
        # Updates the images if update=True
        dm = DataManager(update)
        # Extracts the features again if update=True
        ip = ImageProcessor(update)

        if update:
            color_matrix = ip.colors
            grads_matrix = ip.grads

            self.sim_matrix_colors = []
            self.sim_matrix_grads = []

            # Creating similarity matrices
            for i in range(0, dm.get_num_imgs() - 1):
                # Color similarity matrix
                indexes_colors, dists_colors = k_neighbours(query=color_matrix[i].reshape(1, -1), matrix=color_matrix,
                                                            metric="euclidean", k=kn)

                self.sim_matrix_colors.append(list(zip(indexes_colors, dists_colors)))

                # Gradients similarity matrix
                indexes_grads, dists_grads = k_neighbours(query=grads_matrix[i].reshape(1, -1), matrix=grads_matrix,
                                                          metric="euclidean", k=kn)

                self.sim_matrix_grads.append(list(zip(indexes_grads, dists_grads)))

            # Saving similarity matrices in files
            self.sim_matrix_colors = np.array(self.sim_matrix_colors)
            self.sim_matrix_grads = np.array(self.sim_matrix_grads)
            np.savez('{}.npz'.format("sim_matrix_colors"), color=self.sim_matrix_colors)
            np.savez('{}.npz'.format("sim_matrix_grads"), grad=self.sim_matrix_grads)
        else:
            # Reading similarity matrices from files
            self.sim_matrix_colors = np.load("sim_matrix_colors.npz")["color"]
            self.sim_matrix_grads = np.load("sim_matrix_grads.npz")["grad"]


def k_neighbours(query, matrix, metric="euclidean", k=10):
    dists = pairwise_distances(query, matrix, metric=metric)
    dists = np.squeeze(dists)
    sorted_indexes = np.argsort(dists)
    return sorted_indexes[:k], dists[sorted_indexes[:k]]


sc = SimilarityCalculator(False, kn=5)
print(sc.sim_matrix_colors)
print(sc.sim_matrix_grads)
