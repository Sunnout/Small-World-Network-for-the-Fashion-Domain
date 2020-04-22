import networkx as nx

from DataManager import DataManager as dm

class SmallWorldGraph:

    def __init__(self):
        """Ainda não faz nada"""
        graph = nx.Graph()

        key = 0
        for name in dm.get_img_names():
            # Nodes are tuples of (key, nameOfImage)
            graph.add_node((key,name))
            key += 1

        # Edges are tuples of (key1, key2, distance, feature)
        graph.add_edge()

        print(graph.nodes())
        print(graph.edges())