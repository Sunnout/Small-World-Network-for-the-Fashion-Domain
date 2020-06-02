import os

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from DataManager import DataManager as dm
from ImageProcessor import ImageProcessor as ip
from SimilarityCalculator import SimilarityCalculator


# Directory where we save the images created
RESULTS_DIR = "../Results/"
# Directory where we save the output files
FILES_DIR = "../Files/"


class SmallWorldGraph:
    """
    Documentation: https://networkx.github.io/documentation/stable/reference/index.html

    Functions of networkx for Small World Graphs: https://networkx.github.io/documentation/stable/reference/algorithms/smallworld.html
    random_reference(G[, niter, connectivity, seed]) - Compute a random graph by swapping edges of a given graph.
    lattice_reference(G[, niter, D, …]) - Latticize the given graph by swapping edges.

    If we want to compute them ourselves:
    average_clustering(G[, trials, seed]) - Returns the average clustering coefficient of a graph.
    average_shortest_path_length(G[, weight, method]) - Returns the average shortest path length.

    TODO algoritmo para reduzir o numero de arcos
    """

    def __init__(self, update=False):
        """ Creates a graph with the images as nodes and the edges created according
         to the distance matrix that was previously calculated and stored. """

        self.image_names = dm.get_img_names()
        self.sc = SimilarityCalculator(update)
        self.graph = nx.Graph()

        # Reading distance matrix
        self.dist_matrix = np.load(FILES_DIR + "final_dist_matrix.npz")["dist"]

        num_imgs = dm.get_num_imgs(self)
        # Creating all edges and the corresponding nodes as tuples of (index, image_name)
        for i in range(0, num_imgs):
            for j in range(i + 1, num_imgs):
                self.graph.add_edge((i, self.image_names[i]), (j, self.image_names[j]), distance=self.dist_matrix[i, j])

    @staticmethod
    def show_graph(graph, img_size=0.1, graph_name="graph.pdf"):
        """ Creates a figure showing a given graph (graph) and saves it with a given
         name (graph_name). """

        pos = nx.circular_layout(graph)
        fig = plt.figure(figsize=(20, 20))
        ax = plt.subplot(111)
        ax.set_aspect('equal')

        nx.draw_networkx_edges(graph, pos, ax=ax)

        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        trans = ax.transData.transform
        trans2 = fig.transFigure.inverted().transform
        p2 = img_size / 2.0
        for n in graph:
            xx, yy = trans(pos[n])  # figure coordinates
            xa, ya = trans2((xx, yy))  # axes coordinates
            a = plt.axes([xa - p2, ya - p2, img_size, img_size])
            a.set_aspect('equal')
            a.imshow(dm.get_single_img(n[1]))
            a.axis('off')
        ax.axis('off')

        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)

        plt.savefig(RESULTS_DIR + graph_name, format="pdf")

    @staticmethod
    def draw_path(graph, src, dst, img_size=0.1, graph_name="path.pdf"):
        """ Creates a figure showing the shortest path between two images (src and dst)
         on a given graph (graph). Saves it with a given name (graph_name). The images
         are given as tuples of (index, image_name), like so: (0, "img_00000000.jpg"). """

        pos = nx.kamada_kawai_layout(graph)  # ACHO QUE AQUI ESTA O ERRADO!!!!!!!!!!!
        fig = plt.figure(figsize=(25, 20))
        ax = plt.subplot(111)
        ax.set_aspect('equal')

        nodes = nx.shortest_path(graph, src, dst, weight="distance")
        edges = []
        for i in range(1, len(nodes)):
            edges.append((nodes[i - 1], nodes[i]))

        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=edges)

        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        trans = ax.transData.transform
        trans2 = fig.transFigure.inverted().transform
        p2 = img_size / 2.0
        for n in nodes:
            xx, yy = trans(pos[n])  # Figure coordinates
            xa, ya = trans2((xx, yy))  # Axis coordinates
            a = plt.axes([xa - p2, ya - p2, img_size, img_size])
            a.set_aspect('equal')
            a.imshow(dm.get_single_img(n[1]))
            a.axis('off')
        ax.axis('off')

        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)

        plt.savefig(RESULTS_DIR + graph_name, format="pdf")

    def draw_neighbours(self, src, graph_name="neighbours.pdf", k=10, feat="colors"):
        """ Creates a figure showing the k-NN of a given image (src), according to a
         given feature (feat), side by side. Saves it with a given name (graph_name).
         src is given as a tuple of (index, image_name), like so: (0, "img_00000000.jpg"). """

        idx = dm.get_img_names()
        fig = plt.figure(figsize=(25, 20))
        columns = k
        rows = 1

        if feat == "colors":
            neighbours = self.sc.color_neigh
        elif feat == "grads":
            neighbours = self.sc.grads_neigh
        elif feat == "vgg1":
            neighbours = self.sc.vgg16_block1_neigh
        elif feat == "vgg2":
            neighbours = self.sc.vgg16_block2_neigh
        elif feat == "vgg3":
            neighbours = self.sc.vgg16_block3_neigh
        else:
            neighbours = self.sc.color_neigh

        for i in range(1, columns * rows + 1):
            img = ip.center_crop_image(dm.get_single_img(idx[neighbours[src[0]][i - 1]]))
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
            if i == 1:
                plt.title("Query Image")
            else:
                plt.title("Neighbour " + str(i - 1))
            plt.axis("off")

        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)

        fig.savefig(RESULTS_DIR + graph_name, format="pdf")

    def calc_sw_measure(self):
        """ Calculates small word measure: compares the clustering of a given network
         to an equivalent lattice network and its path length to an equivalent random
         network. """

        return nx.sigma(self.color_graph)

    def calc_small_coefficient(self):
        """ Calculates small-coefficient: compares clustering and path length
         of a given network to an equivalent random network with same degree. """

        return nx.omega(self.color_graph)
