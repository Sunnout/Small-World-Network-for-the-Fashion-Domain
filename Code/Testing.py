from Code.SmallWorldGraph import SmallWorldGraph
from Code.DataManager import DataManager as dm


sw = SmallWorldGraph(False)
sw.show_full_graph(sw.graph)
sw.show_node_neighbours(dm.get_img_index("img_00000000.jpg"), graph_name="0_node_neighbours.pdf");
