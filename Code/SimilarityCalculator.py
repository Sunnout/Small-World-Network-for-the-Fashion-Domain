import os

import numpy as np
from sklearn.metrics import pairwise_distances

from Code.DataManager import DataManager
from Code.ImageProcessor import ImageProcessor, load_feature

from Code.Constants import FILES_DIR, SAMPLE_SET_SIZE, HOC_MATRIX_FILE, HOG_MATRIX_FILE, COLOR_NEIGH_FILE, \
    GRADS_NEIGH_FILE, FINAL_DISTANCES_FILE, KNN, NPZ_EXTENSION, N_NEIGHBOURS, \
    VGG_NEIGH_FILES, VGG_MATRIX_FILES


class SimilarityCalculator:

    def __init__(self, update=False, compute_final_distances=False, layers=[]):
        """ Computes the k-NN of all the images according to all the features. Creates
         a similarity matrix (final_matrix) by computing the pairwise distances over all
         images, according to all features. Then, normalizes those distances and sums them.
         That sum of distances will be the similarity measure stored in the matrix, for each
         pair of images. """

        # Updates the image database if update=True
        self.dm = DataManager(update)
        # Extracts the features again if update=True
        self.ip = ImageProcessor(update, layers)

        num_imgs = self.dm.get_num_imgs()
        if update:
            self.final_matrix = np.zeros((num_imgs, num_imgs))

            # Calculating k-NN of every image according to all features
            self.calculate_neighbours(num_imgs, load_feature(HOC_MATRIX_FILE), COLOR_NEIGH_FILE)
            self.calculate_neighbours(num_imgs, load_feature(HOG_MATRIX_FILE), GRADS_NEIGH_FILE)
            for layer in layers:
                self.calculate_neighbours(num_imgs, load_feature(VGG_MATRIX_FILES[layer]),
                                          VGG_NEIGH_FILES[layer])

            # Calculating pairwise distance of every image according to all features
            self.calculate_final_distance_matrix(num_imgs, layers)

        elif compute_final_distances:
            # Calculating pairwise distance of every image according to all features
            self.calculate_final_distance_matrix(num_imgs, layers)

    def calculate_neighbours(self, num_imgs, feat_matrix, npz_name):
        """ Computes the k-NN of all the images according to a given feature matrix
         (feat_matrix) and stores them in a file with a given name (npz_name). """

        if not os.path.exists(FILES_DIR):
            os.makedirs(FILES_DIR)

        feature_neigh = []
        for i in range(0, num_imgs):
            # Calculating k-NN of image i
            neighbours, _ = self.k_neighbours(feat_matrix[i].reshape(1, -1), feat_matrix, k=N_NEIGHBOURS)
            feature_neigh.append(neighbours)

        # Saving neighbours matrix in file
        np.savez('{}.npz'.format(FILES_DIR + npz_name), knn=feature_neigh)

    def calculate_final_distance_matrix(self, num_imgs, layers):
        self.final_matrix = np.zeros((num_imgs, num_imgs))

        # Calculating pairwise distance of every image according to color feature
        self.calculate_distances(num_imgs, load_feature(HOC_MATRIX_FILE))

        # Calculating pairwise distance of every image according to gradient feature
        self.calculate_distances(num_imgs, load_feature(HOG_MATRIX_FILE))

        # Calculating pairwise distance of every image according to the given VGG16 layers
        for layer in layers:
            self.calculate_distances(num_imgs, load_feature(VGG_MATRIX_FILES[layer]))

        # Saving the same values in opposite indexes because the final distance matrix is symmetric
        for i in range(0, num_imgs):
            for j in range(i + 1, num_imgs):
                self.final_matrix[j, i] = self.final_matrix[i, j]

        if not os.path.exists(FILES_DIR):
            os.makedirs(FILES_DIR)

        # Save the final distances matrix to a file
        np.savez('{}.npz'.format(FILES_DIR + FINAL_DISTANCES_FILE), dist=self.final_matrix)

    def calculate_distances(self, num_imgs, feat_matrix):
        """ Computes the pairwise distances over all images, according to the
         feature matrix. Then, normalizes those distances and sums them to the
         corresponding indexes of the final similarity matrix. """

        # Calculating normalization value
        normalizer = self.calc_sum_distances(self.dm, feat_matrix)

        feature_neigh = []
        for i in range(0, num_imgs):
            for j in range(i + 1, num_imgs):
                # Calculating the distances, normalizing and adding them to the final similarity matrix
                dist = pairwise_distances(feat_matrix[i].reshape(1, -1), feat_matrix[j].reshape(1, -1),
                                          metric="euclidean")
                norm_dist = np.squeeze(dist / normalizer)
                self.final_matrix[i, j] += norm_dist

    @staticmethod
    def k_neighbours(query, matrix, metric="euclidean", k=10):
        """ Computes the k-NN of a given image (query) by calculating the pairwise distance
         of that image and all the images in the given matrix (matrix). Sorts the distances
         and returns the sorted indexes of the neighbours and the sorted distances. """

        dists = pairwise_distances(query, matrix, metric=metric)
        dists = np.squeeze(dists)
        sorted_indexes = np.argsort(dists)
        return sorted_indexes[:k], dists[sorted_indexes[:k]]

    @staticmethod
    def calc_max_distances(dm, feat_matrix, metric="euclidean"):
        """ Gets a sample of size SAMPLE_SET_SIZE from the image database. Calculates the
         pairwise distance over all images according to the given feature matrix (feat_matrix).
         Saves the maximum distance and returns it, so that it can then be used to normalize
         the distances calculated later over the whole database. """

        imgs = dm.get_rand_set(SAMPLE_SET_SIZE)

        max_dist = -1
        for img1 in imgs:
            for img2 in imgs:
                if img1 != img2:
                    d = pairwise_distances(feat_matrix[img1].reshape(1, -1),
                                           feat_matrix[img2].reshape(1, -1),
                                           metric=metric)
                    if d > max_dist:
                        max_dist = d

        return max_dist

    @staticmethod
    def calc_sum_distances(dm, feat_matrix, metric="euclidean"):
        """ Gets a sample of size SAMPLE_SET_SIZE from the image database. Calculates the
         pairwise distance over all images according to the given feature matrix (feat_matrix).
         Sums those distances and returns the sum, so that it can then be used to normalize
         the distances calculated later over the whole database. """

        imgs = dm.get_rand_set(SAMPLE_SET_SIZE)

        sum_dist = 0
        for img1 in imgs:
            for img2 in imgs:
                if img1 != img2:
                    sum_dist += pairwise_distances(feat_matrix[img1].reshape(1, -1),
                                                   feat_matrix[img2].reshape(1, -1),
                                                   metric=metric)

        return sum_dist


def load_neigh(npz_name):
    """ Reads the neighbours matrix from a file with a given name (npz_name). """

    return np.load(FILES_DIR + npz_name + NPZ_EXTENSION, mmap_mode="r")[KNN]
