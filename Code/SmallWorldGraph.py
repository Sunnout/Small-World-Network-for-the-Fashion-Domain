import os

import networkx as nx
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from Code.Constants import FILES_DIR, RESULTS_DIR, FINAL_DISTANCES_FILE, NPZ_EXTENSION, COLOR_NEIGH_FILE, \
    GRADS_NEIGH_FILE, VGG_BLOCK1_NEIGH_FILE, VGG_BLOCK2_NEIGH_FILE, VGG_BLOCK3_NEIGH_FILE, VGG16_BLOCK1_POOL_LAYER, \
    VGG16_BLOCK2_POOL_LAYER, VGG16_BLOCK3_POOL_LAYER, VGG16_BLOCK4_POOL_LAYER, VGG16_BLOCK5_POOL_LAYER, \
    VGG_BLOCK4_NEIGH_FILE, VGG_BLOCK5_NEIGH_FILE, DIST
from Code.DataManager import DataManager as dm
import Code.ImageProcessor as ip
from Code.SimilarityCalculator import SimilarityCalculator, load_neigh


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

    def __init__(self, update=False, layers=[]):
        """ Creates a graph with the images as nodes and the edges created according
         to the distance matrix that was previously calculated and stored. """
        tf.compat.v1.enable_eager_execution()
        self.sc = SimilarityCalculator(update, layers)
        self.graph = nx.Graph()

        # Reading distance matrix
        self.dist_matrix = np.load(FILES_DIR + FINAL_DISTANCES_FILE + NPZ_EXTENSION)[DIST]

        num_imgs = len(dm.get_img_names())
        # Creating all edges and the corresponding nodes as tuples of (index, image_name)
        for i in range(0, num_imgs):
            self.add_kneighbours(i, k=5)

    def add_kneighbours(self, node, k=10):
        sorted_idx = np.argsort(self.dist_matrix[node])
        for j in sorted_idx[1:k + 1]:
            if not self.graph.has_edge((j, self.sc.dm.image_names[j]), (node, self.sc.dm.image_names[node])):
                self.graph.add_edge((node, self.sc.dm.image_names[node]), (j, self.sc.dm.image_names[j]),
                                    distance=self.dist_matrix[node, j])

    @staticmethod
    def show_graph(graph, img_size=0.1, graph_name="graph.pdf"):
        """ Creates a figure showing a given graph (graph) and saves it with a given
         name (graph_name). """

        pos = nx.circular_layout(graph)
        fig = plt.figure(figsize=(35, 35))
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

        pos = nx.circular_layout(graph)
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
            neighbours = load_neigh(COLOR_NEIGH_FILE)
        elif feat == "grads":
            neighbours = load_neigh(GRADS_NEIGH_FILE)
        elif feat == VGG16_BLOCK1_POOL_LAYER:
            neighbours = load_neigh(VGG_BLOCK1_NEIGH_FILE)
        elif feat == VGG16_BLOCK2_POOL_LAYER:
            neighbours = load_neigh(VGG_BLOCK2_NEIGH_FILE)
        elif feat == VGG16_BLOCK3_POOL_LAYER:
            neighbours = load_neigh(VGG_BLOCK3_NEIGH_FILE)
        elif feat == VGG16_BLOCK4_POOL_LAYER:
            neighbours = load_neigh(VGG_BLOCK4_NEIGH_FILE)
        elif feat == VGG16_BLOCK5_POOL_LAYER:
            neighbours = load_neigh(VGG_BLOCK5_NEIGH_FILE)
        else:
            neighbours = load_neigh(COLOR_NEIGH_FILE)

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

        return nx.sigma(self.graph)

    def calc_small_coefficient(self):
        """ Calculates small-coefficient: compares clustering and path length
         of a given network to an equivalent random network with same degree. """

        return nx.omega(self.graph)

    def show_node_neighbours(self, node, img_size=0.1, graph_name="ego_graph.pdf"):
        ego = nx.ego_graph(self.graph, node, undirected=True)
        # self.show_graph(ego, img_size=0.1, graph_name="ego_graph.pdf")
        for (u, v) in ego.edges():
            if u != node and v != node:
                ego.remove_edge(u, v)
        pos = nx.spring_layout(ego)
        fig = plt.figure(figsize=(20, 20))
        ax = plt.subplot(111)
        ax.set_aspect('equal')

        nx.draw_networkx_edges(ego, pos, ax=ax)

        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        trans = ax.transData.transform
        trans2 = fig.transFigure.inverted().transform
        p2 = img_size / 2.0
        for n in ego:
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
    def print_metrics(graph):
        node_count = len(list(graph.nodes))
        edge_count = 0

        # Calculate the average distance of each edge in the graph
        avg_edge_dist = 0.0
        for (u, v, d) in graph.edges.data('distance', default=1.0):
            avg_edge_dist += d
            edge_count += 1
        avg_edge_dist = avg_edge_dist / edge_count

        # Calculate the average edges per node and the average shortest path
        avg_edge_count = 0.0
        avg_shortest_path_size = 0.0
        for node in graph:
            avg_edge_count = len(list(graph.edges(node)))
            for node2 in graph:
                if node[0] != node2[0]:
                    path = nx.shortest_path(graph, node, node2, weight="distance")
                    avg_shortest_path_size += len(list(path))

        avg_edge_count = avg_edge_count / node_count
        avg_shortest_path_size = avg_shortest_path_size / ((node_count * (node_count - 1)) / 2)

        # Calculate the additional metrics like sigma, omega and the clustering coefficient
        sigma = nx.sigma(graph)
        omega = nx.omega(graph)
        clustering_coefficient = nx.average_clustering(graph)

        print("---------- ------ Metrics ------ ----------")
        print("Node Count: \t\t\t\t" + str(node_count))
        print("Edge Count: \t\t\t\t" + str(edge_count))
        print("---------- ---------- ---------- ----------")
        print("Average Edge Count: \t\t" + str(avg_edge_count))
        print("Average Edge Distance: \t" + str(avg_edge_dist))
        print("Average Shortest Path: \t" + str(avg_shortest_path_size))
        print("---------- ---------- ---------- ----------")
        print("Sigma: \t\t\t\t" + str(sigma))
        print("Omega: \t\t\t\t" + str(omega))
        print("Clustering Coefficient: \t" + str(clustering_coefficient))
