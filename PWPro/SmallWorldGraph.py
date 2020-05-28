import time

from numpy import sqrt

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

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
    SmallWorldGraph.show_graph(sw.grad_graph, "grad_graph.pdf",img_size=0.05)

    If we want to compute them ourselves:
    average_clustering(G[, trials, seed]) - Returns the average clustering coefficient of a graph.
    average_shortest_path_length(G[, weight, method]) - Returns the average shortest path length.

    """

    def __init__(self, update=False):

        self.image_names = dm.get_img_names()

        sc = SimilarityCalculator(update)

        # Reading distance matrix
        self.dist_matrix = np.load("final_dist_matrix.npz")["dist"]

        self.graph = nx.Graph()
        added_edges = []
        num_imgs = dm.get_num_imgs(self)
        # Creating all nodes: (index, image_name)
        for i in range(0, num_imgs):
            for j in range(i+1 , num_imgs):
                self.graph.add_edge((i, self.image_names[i]), (j, self.image_names[j]), distance=self.dist_matrix[i, j])


            # Creating all edges for each node: (current_node_index, neighbor_index)
            """for j in range(1, self.number_neighbors):
                # Check if color edge already exists
                idx = (int)(self.sim_matrix_colors[i, j][0])
                if (i, idx, 'c') and (idx, i, 'c') not in added_edges:
                    self.color_graph.add_edge((i, self.image_names[i]), (idx, self.image_names[idx]), tag='c', length=self.sim_matrix_colors[i, j][1])
                    added_edges.append((i, self.sim_matrix_colors[i, j][0], 'c'))

                # Check if gradient edge already exists
                idx = (int)(self.sim_matrix_grads[i, j][0])
                if (i, idx, 'g') and (idx, i, 'g') not in added_edges:
                    self.grad_graph.add_edge((i, self.image_names[i]), (idx, self.image_names[idx]), tag='g', length=self.sim_matrix_grads[i, j][1])
                    added_edges.append((i, self.sim_matrix_grads[i, j][0], 'g'))"""



        #print(self.graph.nodes())

    @staticmethod
    def show_graph(G, graph_name, img_size=0.1, ):
        pos = nx.circular_layout(G)
        fig = plt.figure(figsize=(20, 20))
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
            a.imshow(dm.get_single_img(n[1]))
            a.axis('off')
        ax.axis('off')
        plt.savefig("./Results/" + graph_name,format="pdf")
        plt.show()


    @staticmethod
    def draw_path(graph, src, dst, img_size=0.1, graph_name="path.pdf"):
        nodes = nx.shortest_path(graph, src, dst, weight="distance")
        pos = nx.circular_layout(graph)
        fig = plt.figure(figsize=(20, 20))
        ax = plt.subplot(111)
        ax.set_aspect('equal')
        edges = []
        for i in range(1, len(nodes)):
            edges.append((nodes[i-1],nodes[i]))
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=edges)

        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)

        trans = ax.transData.transform
        trans2 = fig.transFigure.inverted().transform

        p2 = img_size / 2.0
        for n in nodes:
            xx, yy = trans(pos[n])  # figure coordinates
            xa, ya = trans2((xx, yy))  # axes coordinates
            a = plt.axes([xa - p2, ya - p2, img_size, img_size])
            a.set_aspect('equal')
            a.imshow(dm.get_single_img(n[1]))
            a.axis('off')
        ax.axis('off')
        plt.savefig("./Results/" + graph_name, format="pdf")
        plt.show()



    def calc_sw_measure(self): # This bitch slow asf, takes on average 14 secs
        return nx.sigma(self.color_graph)

    def calc_sw_coefficient(self): # This bitch is also slow asf, takes on average 22 secs
        return nx.omega(self.color_graph)

ts = time.time()
sw = SmallWorldGraph(update=True)
#SmallWorldGraph.show_graph(sw.graph, "graph.pdf",img_size=0.05)

print(SmallWorldGraph.draw_path(sw.graph, (0, "img_00000001.jpg"), (1, "img_00000003.jpg")))

"""print("Building Took: ", time.time() - ts)
ts = time.time()
print("Sigma: ",sw.calc_sw_measure(), " Took: ", time.time() - ts)
ts = time.time()
print("Omega: ", sw.calc_sw_coefficient(), " Took: ", time.time() - ts)"""