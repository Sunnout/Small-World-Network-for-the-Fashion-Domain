from Code.SmallWorldGraph import SmallWorldGraph
from DataManager import DataManager as dm


sw = SmallWorldGraph(False)
SmallWorldGraph.show_graph(sw.graph)
'''sw.draw_neighbours(dm.get_img_index("img_00000000.jpg"), graph_name="img00_block1.pdf", feat="vgg1")
sw.draw_neighbours(dm.get_img_index("img_00000000.jpg"), graph_name="img00_block2.pdf", feat="vgg2")
sw.draw_neighbours(dm.get_img_index("img_00000000.jpg"), graph_name="img00_block3.pdf", feat="vgg3")'''

