import networkx as nx
import numpy as np

from DataManager import DataManager as dm
from SimilarityCalculator import SimilarityCalculator


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
        added_edges = []

        # Creating all nodes: (index, image_name)
        for i in range(0, dm.get_num_imgs(self)):
            self.color_graph.add_node((i, self.image_names[i]))

            # Creating all edges for each node: (current_node_index, neighbor_index)
            for j in range(1, self.number_neighbors):
                # Check if color edge already exists
                if (i, self.sim_matrix_colors[i, j][0], 'c') and (self.sim_matrix_colors[i, j][0], i, 'c') not in added_edges:
                    self.color_graph.add_edge(i, self.sim_matrix_colors[i, j][0], tag='c')
                    added_edges.append((i, self.sim_matrix_colors[i, j][0], 'c'))

                # Check if gradient edge already exists
                if (i, self.sim_matrix_grads[i, j][0], 'g') and (self.sim_matrix_grads[i, j][0], i, 'g') not in added_edges:
                    self.color_graph.add_edge(i, self.sim_matrix_grads[i, j][0], tag='g')
                    added_edges.append((i, self.sim_matrix_grads[i, j][0], 'g'))

        print(self.color_graph.edges.data())


sw = SmallWorldGraph(update=False)
#print(sw.color_graph.has_edge(*(0, 9, {'tag': 'g'})[:2]))
#print(sw.color_graph[0][1][0]['tag'])

