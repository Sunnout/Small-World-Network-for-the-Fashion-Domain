import numpy as np
from sklearn.metrics import pairwise_distances

from DataManager import DataManager
from ImageProcessor import ImageProcessor

# Size of the sample set use to obtain the normalization values
SAMPLE_SET_SIZE = 10
# Directory where we save the output files
FILES_DIR = "../Files/"


class SimilarityCalculator:

    def __init__(self, update=False):
        """ Computes the k-NN of all the images according to all the features. Creates
         a similarity matrix (final_matrix) by computing the pairwise distances over all
         images, according to all features. Then, normalizes those distances and sums them.
         That sum of distances will be the similarity measure stored in the matrix, for each
         pair of images. """

        # Updates the image database if update=True
        dm = DataManager(update)
        # Extracts the features again if update=True
        ip = ImageProcessor(update)

        self.color_matrix = ip.colors
        self.grads_matrix = ip.grads
        self.vgg16_block1_matrix = ip.vgg_block1
        self.vgg16_block2_matrix = ip.vgg_block2
        self.vgg16_block3_matrix = ip.vgg_block3

        if update:
            color_normalizer, grad_normalizer = self.calc_sum_distances(dm)
            num_imgs = dm.get_num_imgs()
            self.final_matrix = np.zeros((num_imgs, num_imgs))

            for i in range(0, num_imgs):
                # Calculating k-NN of image i according to color feature
                neighbours, _ = self.k_neighbours(self.color_matrix[i].reshape(1, -1), self.color_matrix, k=10)
                self.color_neigh.append(neighbours)

                # Calculating k-NN of image i according to gradient feature
                neighbours, _ = self.k_neighbours(self.grads_matrix[i].reshape(1, -1), self.grads_matrix, k=10)
                self.grads_neigh.append(neighbours)

                # Calculating k-NN of image i according to vgg16_block1 feature
                neighbours, _ = self.k_neighbours(self.vgg16_block1_matrix[i].reshape(1, -1), self.vgg16_block1_matrix,
                                                  k=10)
                self.vgg16_block1_neigh.append(neighbours)

                # Calculating k-NN of image i according to vgg16_block2 feature
                neighbours, _ = self.k_neighbours(self.vgg16_block2_matrix[i].reshape(1, -1), self.vgg16_block2_matrix,
                                                  k=10)
                self.vgg16_block2_neigh.append(neighbours)

                # Calculating k-NN of image i according to vgg16_block3 feature
                neighbours, _ = self.k_neighbours(self.vgg16_block3_matrix[i].reshape(1, -1), self.vgg16_block3_matrix,
                                                  k=10)
                self.vgg16_block3_neigh.append(neighbours)

                for j in range(i + 1, num_imgs):
                    # Calculating color distances, normalizing and adding them to the final matrix
                    dist = pairwise_distances(self.color_matrix[i].reshape(1, -1), self.color_matrix[j].reshape(1, -1),
                                              metric="euclidean")
                    norm_dist = np.squeeze(dist/color_normalizer)
                    self.final_matrix[i, j] += norm_dist

                    # Calculating gradient distances, normalizing and adding them to the final matrix
                    dist = pairwise_distances(self.grads_matrix[i].reshape(1, -1), self.grads_matrix[j].reshape(1, -1),
                                              metric="euclidean")
                    norm_dist = np.squeeze(dist/grad_normalizer)
                    self.final_matrix[i, j] += norm_dist

                    # Calculating vgg16_block1 distances, normalizing and adding them to the final matrix
                    # TODO

                    # Calculating vgg16_block2 distances, normalizing and adding them to the final matrix
                    # TODO

                    # Calculating vgg16_block3 distances, normalizing and adding them to the final matrix
                    # TODO

                    # Saving the same values in opposite indexes because matrix is symmetric
                    self.final_matrix[j, i] = self.final_matrix[i, j]

            # Saving distance matrix in file
            np.savez('{}.npz'.format(FILES_DIR + "final_dist_matrix"), dist=self.final_matrix)
            np.savez('{}.npz'.format(FILES_DIR + "color_neighbours"), knn=self.color_neigh)
            np.savez('{}.npz'.format(FILES_DIR + "grads_neighbours"), knn=self.grads_neigh)
            np.savez('{}.npz'.format(FILES_DIR + "vgg16_block1_neighbours"), knn=self.vgg16_block1_neigh)
            np.savez('{}.npz'.format(FILES_DIR + "vgg16_block2_neighbours"), knn=self.vgg16_block2_neigh)
            np.savez('{}.npz'.format(FILES_DIR + "vgg16_block3_neighbours"), knn=self.vgg16_block3_neigh)
        else:
            # Reading distance matrix from file
            self.final_matrix = np.load(FILES_DIR + "final_dist_matrix.npz")["dist"]
            self.color_neigh = np.load(FILES_DIR + "color_neighbours.npz")["knn"]
            self.grads_neigh = np.load(FILES_DIR + "grads_neighbours.npz")["knn"]
            self.vgg16_block1_neigh = np.load(FILES_DIR + "vgg16_block1_neighbours.npz")["knn"]
            self.vgg16_block2_neigh = np.load(FILES_DIR + "vgg16_block2_neighbours.npz")["knn"]
            self.vgg16_block3_neigh = np.load(FILES_DIR + "vgg16_block3_neighbours.npz")["knn"]

    @staticmethod
    def k_neighbours(query, matrix, metric="euclidean", k=10):
        """ Computes the k-NN of a given image (query) by calculating the pairwise distance
         of that image and all the images in the given matrix (matrix). Sorts the distances
         and returns the sorted indexes of the neighbours and the sorted distances. """

        dists = pairwise_distances(query, matrix, metric=metric)
        dists = np.squeeze(dists)
        sorted_indexes = np.argsort(dists)
        return sorted_indexes[:k], dists[sorted_indexes[:k]]

    def calc_max_distances(self, dm, metric="euclidean"):
        """ Gets a sample of size SAMPLE_SET_SIZE from the image database. Calculates the
         pairwise distance according to all the features, over all the images, and saves
         the maximum distances. Returns these maximum distances, that can then be used to
         normalize the distances calculated over the whole database. """

        imgs = dm.get_rand_set(SAMPLE_SET_SIZE)

        max_dist_color = -1
        max_dist_grad = -1
        for img1 in imgs:
            for img2 in imgs:
                if img1 != img2:
                    d = pairwise_distances(self.color_matrix[img1].reshape(1, -1),
                                           self.color_matrix[img2].reshape(1, -1),
                                           metric=metric)
                    if d > max_dist_color:
                        max_dist_color = d

                    d = pairwise_distances(self.grads_matrix[img1].reshape(1, -1),
                                           self.grads_matrix[img2].reshape(1, -1),
                                           metric=metric)
                    if d > max_dist_grad:
                        max_dist_grad = d

                    # TODO vgg16 features max distances

        return max_dist_color, max_dist_grad

    def calc_sum_distances(self, dm, metric="euclidean"):
        """ Gets a sample of size SAMPLE_SET_SIZE from the image database. Calculates the
         pairwise distance according to all the features, over all the images, and sums
         those distances. Returns these distance sums, that can then be used to normalize
         the distances calculated over the whole database. """

        imgs = dm.get_rand_set(SAMPLE_SET_SIZE)

        sum_dist_color = 0
        sum_dist_grad = 0
        for img1 in imgs:
            for img2 in imgs:
                if img1 != img2:
                    sum_dist_color += pairwise_distances(self.color_matrix[img1].reshape(1, -1),
                                                         self.color_matrix[img2].reshape(1, -1),
                                                         metric=metric)

                    sum_dist_grad += pairwise_distances(self.grads_matrix[img1].reshape(1, -1),
                                                        self.grads_matrix[img2].reshape(1, -1),
                                                        metric=metric)

                    # TODO vgg16 features sum distances

        return sum_dist_color, sum_dist_grad
