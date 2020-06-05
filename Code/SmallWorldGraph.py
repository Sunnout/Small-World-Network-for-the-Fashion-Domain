import os

import networkx as nx
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from Code.DataManager import DataManager as dm
import Code.ImageProcessor as ip
from Code.SimilarityCalculator import SimilarityCalculator, load_neigh

from Code.Constants import FILES_DIR, RESULTS_DIR, FINAL_DISTANCES_FILE, NPZ_EXTENSION, COLOR_NEIGH_FILE, \
    GRADS_NEIGH_FILE, VGG_BLOCK2_NEIGH_FILE, VGG_BLOCK3_NEIGH_FILE, \
    VGG16_BLOCK2_POOL_LAYER, VGG16_BLOCK3_POOL_LAYER, VGG16_BLOCK4_POOL_LAYER, VGG16_BLOCK5_POOL_LAYER, \
    VGG_BLOCK4_NEIGH_FILE, VGG_BLOCK5_NEIGH_FILE, DIST, EDGES


class SmallWorldGraph:

    def __init__(self, update=False, compute_final_distances=False, layers=[]):
        """ Creates a graph with the images as nodes and the edges created according
         to the distance matrix that was previously calculated and stored. """

        # To be able to extract numpy arrays from Tensor Objects
        tf.compat.v1.enable_eager_execution()

        self.sc = SimilarityCalculator(update, compute_final_distances, layers)
        self.graph = nx.Graph()

        # Reading distance matrix from file
        self.dist_matrix = np.load(FILES_DIR + FINAL_DISTANCES_FILE + NPZ_EXTENSION)[DIST]

        # Creating nodes as tuples of (index, image_name), for all the images in the database
        node_idx = 0
        for name in self.sc.dm.get_img_names():
            self.graph.add_node((node_idx, name))
            node_idx += 1

        # Creating the edges for node with index i
        for i in range(0, self.sc.dm.get_num_imgs()):
            self.add_node_edges(i, n_edges=EDGES)

    def add_node_edges(self, node, n_edges=10):
        """ Given a node index (node) and the number of edges to create (n_edges),
         creates edges from that node to the nodes closest it, according to the
         final distance matrix. """

        sorted_idx = np.argsort(self.dist_matrix[node])
        for j in sorted_idx[1:n_edges + 1]:
            name_j = self.sc.dm.image_names[j]
            name_node = self.sc.dm.image_names[node]
            if not self.graph.has_edge((j, name_j), (node, name_node)):
                self.graph.add_edge((node, name_node), (j, name_j), distance=self.dist_matrix[node, j])

    def show_full_graph(self, img_size=0.1, graph_name="graph.pdf"):
        """ Creates a figure showing the full graph and saves it with a given name
         (graph_name). """

        pos = nx.circular_layout(self.graph)
        fig = plt.figure(figsize=(35, 35))
        ax = plt.subplot(111)
        ax.set_aspect('equal')

        nx.draw_networkx_edges(self.graph, pos, ax=ax)

        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        trans = ax.transData.transform
        trans2 = fig.transFigure.inverted().transform
        p2 = img_size / 2.0
        for n in self.graph:
            xx, yy = trans(pos[n])
            xa, ya = trans2((xx, yy))
            a = plt.axes([xa - p2, ya - p2, img_size, img_size])
            a.set_aspect('equal')
            a.imshow(dm.get_single_img(n[1]))
            a.axis('off')
        ax.axis('off')

        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)

        plt.savefig(RESULTS_DIR + graph_name, format="pdf")

    def show_shortest_path(self, src, dst, img_size=0.1, graph_name="path.pdf"):
        """ Creates a figure showing the shortest path between two images (src and dst)
         and saves it with a given name (graph_name). The images are given as tuples of
         (index, image_name), like so: (0, "img_00000000.jpg"). """

        pos = nx.circular_layout(self.graph)
        fig = plt.figure(figsize=(25, 20))
        ax = plt.subplot(111)
        ax.set_aspect('equal')

        nodes = nx.shortest_path(self.graph, src, dst, weight="distance")
        edges = []
        for i in range(1, len(nodes)):
            edges.append((nodes[i - 1], nodes[i]))

        nx.draw_networkx_edges(self.graph, pos, ax=ax, edgelist=edges)

        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        trans = ax.transData.transform
        trans2 = fig.transFigure.inverted().transform
        p2 = img_size / 2.0
        for n in nodes:
            xx, yy = trans(pos[n])
            xa, ya = trans2((xx, yy))
            a = plt.axes([xa - p2, ya - p2, img_size, img_size])
            a.set_aspect('equal')
            a.imshow(dm.get_single_img(n[1]))
            a.axis('off')
        ax.axis('off')

        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)

        plt.savefig(RESULTS_DIR + graph_name, format="pdf")

    def show_node_neighbours(self, node, img_size=0.1, graph_name="node_neighbours.pdf"):
        """ Creates a figure showing the neighbours of a given image (node) in the graph.
         Saves it with a given name (graph_name). Node is given as a tuple of (index, image_name),
         like so: (0, "img_00000000.jpg"). """

        ego = nx.ego_graph(self.graph, node, undirected=True)
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
            xx, yy = trans(pos[n])
            xa, ya = trans2((xx, yy))
            a = plt.axes([xa - p2, ya - p2, img_size, img_size])
            a.set_aspect('equal')
            a.imshow(dm.get_single_img(n[1]))
            a.axis('off')
        ax.axis('off')

        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)

        plt.savefig(RESULTS_DIR + graph_name, format="pdf")

    def print_graph_metrics(self):
        """ Computes several metrics for the graph. """

        # Calculate number of nodes and edges
        node_count = len(list(self.graph.nodes))
        edge_count = len(list(self.graph.edges))

        # Calculate average edge distance
        avg_edge_dist = 0.0
        for (u, v, d) in self.graph.edges.data('distance', default=1.0):
            avg_edge_dist += d
        avg_edge_dist = avg_edge_dist / edge_count

        # Calculate average edges per node
        avg_edge_count = 0.0
        for node in self.graph:
            avg_edge_count += len(list(self.graph.edges(node)))
        avg_edge_count = avg_edge_count / node_count

        # Calculate sigma, omega
        sigma = nx.sigma(self.graph)
        omega = nx.omega(self.graph)

        # Calculate clustering coefficient, average shortest path length and connectivity
        clustering_coefficient = nx.average_clustering(self.graph)
        avg_shortest_path_len = nx.average_shortest_path_length(self.graph)
        is_connected = nx.component.is_connected(self.graph)

        print("---------- ------ Metrics ------ ----------")
        print("Number of Nodes: \t\t\t\t" + str(node_count))
        print("Number of Edges: \t\t\t\t" + str(edge_count))
        print("---------- ---------- ---------- ----------")
        print("Connected Graph: \t\t\t\t" + str(is_connected))
        print("---------- ---------- ---------- ----------")
        print("Average Neighbour Count: \t\t" + str(avg_edge_count))
        print("Average Neighbour Distance: \t" + str(avg_edge_dist))
        print("Average Shortest Path Length: \t" + str(avg_shortest_path_len))
        print("Average Clustering Coefficient: \t" + str(clustering_coefficient))
        print("---------- ---------- ---------- ----------")
        print("Small World Measure: \t\t\t\t" + str(sigma))
        print("Small Coefficient: \t\t\t\t" + str(omega))

    @staticmethod
    def show_feature_neighbours(src, graph_name="feature_neighbours.pdf", k=10, feat="colors"):
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
