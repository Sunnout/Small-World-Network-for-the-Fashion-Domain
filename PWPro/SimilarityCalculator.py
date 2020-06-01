import numpy as np

from sklearn.metrics import pairwise_distances
from DataManager import DataManager
from ImageProcessor import ImageProcessor


class SimilarityCalculator:

    def __init__(self, update=False):
        self.color_matrix = []
        self.grads_matrix = []
        self.vgg16_matrix = []
        self.color_neigh = []
        self.grads_neigh = []
        self.vgg16_neigh = []
        self.final_matrix = []

        # Updates the images if update=True
        dm = DataManager(update)
        # Extracts the features again if update=True
        ip = ImageProcessor(update)

        self.color_matrix = ip.colors
        self.grads_matrix = ip.grads
        self.vgg16_block1_matrix = ip.vgg_block1
        self.vgg16_block2_matrix = ip.vgg_block2
        self.color_neigh = []
        self.grads_neigh = []
        self.vgg16_block1_neigh = []
        self.vgg16_block2_neigh = []

        if update:

            color_max, grad_max = self.calc_sum_distances(dm)
            num_imgs = dm.get_num_imgs()

            self.final_matrix = np.zeros((num_imgs, num_imgs))

            # Creating similarity matrices
            for i in range(0, num_imgs):
                neighbours, _ = self.k_neighbours(self.color_matrix[i].reshape(1, -1), self.color_matrix, k=10)
                self.color_neigh.append(neighbours)
                neighbours, _ = self.k_neighbours(self.grads_matrix[i].reshape(1, -1), self.grads_matrix, k=10)
                self.grads_neigh.append(neighbours)
                neighbours, _ = self.k_neighbours(self.vgg16_block1_matrix[i].reshape(1, -1), self.vgg16_block1_matrix, k=10)
                self.vgg16_block1_neigh.append(neighbours)
                neighbours, _ = self.k_neighbours(self.vgg16_block2_matrix[i].reshape(1, -1), self.vgg16_block2_matrix, k=10)
                self.vgg16_block2_neigh.append(neighbours)

                # Color similarity matrix
                for j in range(i + 1, num_imgs):
                    dist = pairwise_distances(self.color_matrix[i].reshape(1, -1), self.color_matrix[j].reshape(1, -1),
                                              metric="euclidean")
                    norm_dist = np.squeeze(dist / color_max)
                    self.final_matrix[i, j] += norm_dist

                    dist = pairwise_distances(self.grads_matrix[i].reshape(1, -1), self.grads_matrix[j].reshape(1, -1),
                                              metric="euclidean")
                    norm_dist = np.squeeze(dist / grad_max)
                    self.final_matrix[i, j] += norm_dist
                    self.final_matrix[j, i] = self.final_matrix[i, j]

            # Saving distance matrix in file
            np.savez('{}.npz'.format("final_dist_matrix"), dist=self.final_matrix)
            np.savez('{}.npz'.format("color_neighbours"), knn=self.color_neigh)
            np.savez('{}.npz'.format("grads_neighbours"), knn=self.grads_neigh)
            np.savez('{}.npz'.format("vgg16_block1_neighbours"), knn=self.vgg16_block1_neigh)
            np.savez('{}.npz'.format("vgg16_block2_neighbours"), knn=self.vgg16_block2_neigh)
        else:
            # Reading distance matrix from file
            self.final_matrix = np.load("final_dist_matrix.npz")["dist"]
            self.color_neigh = np.load("color_neighbours.npz")["knn"]
            self.grads_neigh = np.load("grads_neighbours.npz")["knn"]
            self.vgg16_block1_neigh = np.load("vgg16_block1_neighbours.npz")["knn"]
            self.vgg16_block2_neigh = np.load("vgg16_block2_neighbours.npz")["knn"]


    def initialize(self, update=False):
        # Updates the images if update=True
        dm = DataManager.initialize(update)
        # Extracts the features again if update=True
        ip = ImageProcessor.initialize(update)

        self.color_matrix = ip.colors
        self.grads_matrix = ip.grads
        self.vgg16_matrix = ip.vgg_block1
        self.color_neigh = []
        self.grads_neigh = []
        self.vgg16_neigh = []

        if update:

            color_max, grad_max = self.calc_sum_distances(dm)
            num_imgs = dm.get_num_imgs()

            self.final_matrix = np.zeros((num_imgs, num_imgs))

            # Creating similarity matrices
            for i in range(0, num_imgs):
                neighbours, _ = self.k_neighbours(self.color_matrix[i].reshape(1, -1), self.color_matrix, k=10)
                self.color_neigh.append(neighbours)
                neighbours, _ = self.k_neighbours(self.grads_matrix[i].reshape(1, -1), self.grads_matrix, k=10)
                self.grads_neigh.append(neighbours)
                neighbours, _ = self.k_neighbours(self.vgg16_matrix[i].reshape(1, -1), self.vgg16_matrix, k=10)
                self.vgg16_neigh.append(neighbours)

                # Color similarity matrix
                for j in range(i + 1, num_imgs):
                    dist = pairwise_distances(self.color_matrix[i].reshape(1, -1), self.color_matrix[j].reshape(1, -1),
                                              metric="euclidean")
                    norm_dist = np.squeeze(dist / color_max)
                    self.final_matrix[i, j] += norm_dist

                    dist = pairwise_distances(self.grads_matrix[i].reshape(1, -1), self.grads_matrix[j].reshape(1, -1),
                                              metric="euclidean")
                    norm_dist = np.squeeze(dist / grad_max)
                    self.final_matrix[i, j] += norm_dist
                    self.final_matrix[j, i] = self.final_matrix[i, j]

            # Saving distance matrix in file
            np.savez('{}.npz'.format("final_dist_matrix"), dist=self.final_matrix)
            np.savez('{}.npz'.format("color_neighbours"), knn=self.color_neigh)
            np.savez('{}.npz'.format("grads_neighbours"), knn=self.grads_neigh)
        else:
            # Reading distance matrix from file
            self.final_matrix = np.load("final_dist_matrix.npz")["dist"]
            self.color_neigh = np.load("color_neighbours.npz")["knn"]
            self.grads_neigh = np.load("grads_neighbours.npz")["knn"]

    @staticmethod
    def k_neighbours(query, matrix, metric="euclidean", k=10):
        dists = pairwise_distances(query, matrix, metric=metric)
        dists = np.squeeze(dists)
        sorted_indexes = np.argsort(dists)
        return sorted_indexes[:k], dists[sorted_indexes[:k]]

    def calc_max_distances(self, dm):
        imgs = dm.get_rand_set(10)
        max_dist_color = -1
        max_dist_grad = -1
        for img1 in imgs:
            for img2 in imgs:
                if img1 != img2:
                    d = pairwise_distances(self.color_matrix[img1].reshape(1, -1),
                                           self.color_matrix[img2].reshape(1, -1),
                                           metric="euclidean")
                    if d > max_dist_color:
                        max_dist_color = d
                    d = pairwise_distances(self.grads_matrix[img1].reshape(1, -1),
                                           self.grads_matrix[img2].reshape(1, -1),
                                           metric="euclidean")
                    if d > max_dist_grad:
                        max_dist_grad = d
        return max_dist_color, max_dist_grad

    def calc_sum_distances(self, dm):
        imgs = dm.get_rand_set(10)
        sum_dist_color = 0
        sum_dist_grad = 0
        for img1 in imgs:
            for img2 in imgs:
                if img1 != img2:
                    sum_dist_color += pairwise_distances(self.color_matrix[img1].reshape(1, -1),
                                           self.color_matrix[img2].reshape(1, -1),
                                           metric="euclidean")

                    sum_dist_grad += pairwise_distances(self.grads_matrix[img1].reshape(1, -1),
                                           self.grads_matrix[img2].reshape(1, -1),
                                           metric="euclidean")
        return sum_dist_color, sum_dist_grad