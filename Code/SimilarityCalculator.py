import os

import numpy as np
from sklearn.metrics import pairwise_distances

from Code.Constants import FILES_DIR, SAMPLE_SET_SIZE, HOC_MATRIX_FILE, HOG_MATRIX_FILE, COLOR_NEIGH_FILE, \
    GRADS_NEIGH_FILE, VGG_BLOCK1_MATRIX_FILE, VGG_BLOCK1_NEIGH_FILE, VGG_BLOCK2_MATRIX_FILE, VGG_BLOCK2_NEIGH_FILE, \
    VGG_BLOCK3_MATRIX_FILE, VGG_BLOCK3_NEIGH_FILE, FINAL_DISTANCES_FILE, KNN, NPZ_EXTENSION
from Code.DataManager import DataManager
from Code.ImageProcessor import ImageProcessor, load_feature


class SimilarityCalculator:

    def __init__(self, update=False):
        """ Computes the k-NN of all the images according to all the features. Creates
         a similarity matrix (final_matrix) by computing the pairwise distances over all
         images, according to all features. Then, normalizes those distances and sums them.
         That sum of distances will be the similarity measure stored in the matrix, for each
         pair of images. """

        # Updates the image database if update=True
        self.dm = DataManager(update)
        # Extracts the features again if update=True
        self.ip = ImageProcessor(update)

        if update:
            num_imgs = self.dm.get_num_imgs()
            self.final_matrix = np.zeros((num_imgs, num_imgs))

            if not os.path.exists(FILES_DIR):
                os.makedirs(FILES_DIR)

            # Calculating k-NN of every image according to color feature
            self.calculate_feature_distances(num_imgs, load_feature(HOC_MATRIX_FILE), COLOR_NEIGH_FILE)

            # Calculating k-NN of every image according to gradient feature
            self.calculate_feature_distances(num_imgs, load_feature(HOG_MATRIX_FILE), GRADS_NEIGH_FILE)

            # Calculating k-NN of image i according to vgg16_block1 feature
            self.calculate_vgg_neighbours(num_imgs, load_feature(VGG_BLOCK1_MATRIX_FILE), VGG_BLOCK1_NEIGH_FILE)

            # Calculating k-NN of image i according to vgg16_block2 feature
            self.calculate_vgg_neighbours(num_imgs, load_feature(VGG_BLOCK2_MATRIX_FILE), VGG_BLOCK2_NEIGH_FILE)

            # Calculating k-NN of image i according to vgg16_block3 feature
            self.calculate_vgg_neighbours(num_imgs, load_feature(VGG_BLOCK3_MATRIX_FILE), VGG_BLOCK3_NEIGH_FILE)

            # Calculating vgg16_block1 distances, normalizing and adding them to the final matrix
            # TODO

            # Calculating vgg16_block2 distances, normalizing and adding them to the final matrix
            # TODO

            # Calculating vgg16_block3 distances, normalizing and adding them to the final matrix
            # TODO

            for i in range(0, num_imgs):
                for j in range(i + 1, num_imgs):
                    # Saving the same values in opposite indexes because matrix is symmetric
                    self.final_matrix[j, i] = self.final_matrix[i, j]

            # Save the calculated distances to an npz file
            np.savez('{}.npz'.format(FILES_DIR + FINAL_DISTANCES_FILE), dist=self.final_matrix)

    def calculate_feature_distances(self, num_imgs, features_matrix, npz_name):
        normalizer = self.calc_sum_distances(self.dm, features_matrix)
        feature_neigh = []
        for i in range(0, num_imgs):
            # Calculating k-NN of image i according to the feature
            neighbours, _ = self.k_neighbours(features_matrix[i].reshape(1, -1), features_matrix, k=10)
            feature_neigh.append(neighbours)
            for j in range(i + 1, num_imgs):
                # Calculating the distances, normalizing and adding them to the final matrix
                dist = pairwise_distances(features_matrix[i].reshape(1, -1), features_matrix[j].reshape(1, -1),
                                          metric="euclidean")
                norm_dist = np.squeeze(dist / normalizer)
                self.final_matrix[i, j] += norm_dist

        # Saving distance matrix in file
        np.savez('{}.npz'.format(FILES_DIR + npz_name), knn=feature_neigh)

    def calculate_vgg_neighbours(self, num_imgs, vgg_block_matrix, npz_name):
        feature_neigh = []
        for i in range(0, num_imgs):
            # Calculating k-NN of image i according to vgg16_blockX feature
            neighbours, _ = self.k_neighbours(vgg_block_matrix[i].reshape(1, -1), vgg_block_matrix, k=10)
            feature_neigh.append(neighbours)

        # Saving distance matrix in file
        np.savez('{}.npz'.format(FILES_DIR + npz_name), knn=feature_neigh)

    @staticmethod
    def k_neighbours(query, matrix, metric="euclidean", k=10):
        """ Computes the k-NN of a given image (query) by calculating the pairwise distance
         of that image and all the images in the given matrix (matrix). Sorts the distances
         and returns the sorted indexes of the neighbours and the sorted distances. """

        dists = pairwise_distances(query, matrix, metric=metric)
        dists = np.squeeze(dists)
        sorted_indexes = np.argsort(dists)
        return sorted_indexes[:k], dists[sorted_indexes[:k]]

    def calc_max_distances(self, dm, feat_matrix, metric="euclidean"):
        """ Gets a sample of size SAMPLE_SET_SIZE from the image database. Calculates the
         pairwise distance according to all the features, over all the images, and saves
         the maximum distances. Returns these maximum distances, that can then be used to
         normalize the distances calculated over the whole database. """

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

                    # TODO vgg16 features max distances

        return max_dist

    def calc_sum_distances(self, dm, feat_matrix, metric="euclidean"):
        """ Gets a sample of size SAMPLE_SET_SIZE from the image database. Calculates the
         pairwise distance according to all the features, over all the images, and sums
         those distances. Returns these distance sums, that can then be used to normalize
         the distances calculated over the whole database. """

        imgs = dm.get_rand_set(SAMPLE_SET_SIZE)

        sum_dist = 0
        for img1 in imgs:
            for img2 in imgs:
                if img1 != img2:
                    sum_dist += pairwise_distances(feat_matrix[img1].reshape(1, -1),
                                                         feat_matrix[img2].reshape(1, -1),
                                                         metric=metric)

                    # TODO vgg16 features sum distances

        return sum_dist


def load_neigh(npz_name):
    # Fetch neighbours matrix from the mentioned file
    return np.load(FILES_DIR + npz_name + NPZ_EXTENSION, mmap_mode="r")[KNN]
