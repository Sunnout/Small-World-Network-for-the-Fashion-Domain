from numpy import sqrt

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from DataManager import DataManager as dm
from SimilarityCalculator import SimilarityCalculator

from PWPro.DataManager import DataManager


class SmallWorldGraph:
    """
    Documentation: https://networkx.github.io/documentation/stable/reference/index.html

    Functions of networkx for Small World Graphs: https://networkx.github.io/documentation/stable/reference/algorithms/smallworld.html
    random_reference(G[, niter, connectivity, seed]) - Compute a random graph by swapping edges of a given graph.
    lattice_reference(G[, niter, D, …]) - Latticize the given graph by swapping edges.
    sigma(G[, niter, nrand, seed]) - Returns the small-world coefficient (sigma) of the given graph.
    omega(G[, niter, nrand, seed]) - Returns the small-world coefficient (omega) of a graph.

    If we want to compute them ourselves:
    average_clustering(G[, trials, seed]) - Returns the average clustering coefficient of a graph.
    average_shortest_path_length(G[, weight, method]) - Returns the average shortest path length.

    """

    def __init__(self, update=False):

        self.image_names = dm.get_img_names()

        sc = SimilarityCalculator(update, kn=5)
        self.number_neighbors = sc.kn

        # Reading similarity matrices
        self.sim_matrix_colors = np.load("sim_matrix_colors.npz")["color"]
        self.sim_matrix_grads = np.load("sim_matrix_grads.npz")["grad"]

        self.color_graph = nx.MultiGraph()
        self.grad_graph = nx.MultiGraph()
        added_edges = []

        # Creating all nodes: (index, image_name)
        for i in range(0, dm.get_num_imgs(self)):
            self.color_graph.add_node((i, self.image_names[i]))
            self.grad_graph.add_node((i, self.image_names[i]))

            # Creating all edges for each node: (current_node_index, neighbor_index)
            for j in range(1, self.number_neighbors):
                # Check if color edge already exists
                if (i, self.sim_matrix_colors[i, j][0], 'c') and (self.sim_matrix_colors[i, j][0], i, 'c') not in added_edges:
                    idx = (int)(self.sim_matrix_colors[i, j][0])
                    self.color_graph.add_edge((i, self.image_names[i]), (idx, self.image_names[idx]), tag='c')
                    added_edges.append((i, self.sim_matrix_colors[i, j][0], 'c'))

                # Check if gradient edge already exists
                if (i, self.sim_matrix_grads[i, j][0], 'g') and (self.sim_matrix_grads[i, j][0], i, 'g') not in added_edges:
                    idx = (int)(self.sim_matrix_grads[i, j][0])
                    self.grad_graph.add_edge((i, self.image_names[i]), (idx, self.image_names[idx]), tag='g')
                    added_edges.append((i, self.sim_matrix_grads[i, j][0], 'g'))

        # print(self.color_graph.edges.data())

    @staticmethod
    def show_graph(G, graph_name, img_size=0.1):
        pos = nx.circular_layout(G)
        fig = plt.figure(figsize=(5, 5))
        ax = plt.subplot(111)
        ax.set_aspect('equal')
        nx.draw_networkx_edges(G, pos, ax=ax)

        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)

        trans = ax.transData.transform
        trans2 = fig.transFigure.inverted().transform

        p2 = img_size / 2.0
        for n in G:
            xx, yy = trans(pos[n])  # figure coordinates
            xa, ya = trans2((xx, yy))  # axes coordinates
            a = plt.axes([xa - p2, ya - p2, img_size, img_size])
            a.set_aspect('equal')
            a.imshow(DataManager.get_single_img(n[1]))
            a.axis('off')
        ax.axis('off')
        plt.savefig(graph_name)
        plt.show()

sw = SmallWorldGraph(update=False)
#print(sw.color_graph.has_edge(*(0, 9, {'tag': 'g'})[:2]))
#print(sw.color_graph[0][1][0]['tag'])

#print("Nodes of graph: ")
nodes = list(sw.color_graph.nodes)
print([i[0] for i in nodes])

# ***** DRAW ****** #

SmallWorldGraph.show_graph(sw.color_graph, "color_graph.png")
SmallWorldGraph.show_graph(sw.grad_graph, "grad_graph.png")



