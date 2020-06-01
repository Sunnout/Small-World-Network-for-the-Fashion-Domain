from PWPro.SmallWorldGraph import SmallWorldGraph
from PWPro.DataManager import DataManager as dm


sw = SmallWorldGraph(False)
sw.draw_kneighbours(dm.get_img_index("img_00000000.jpg"), graph_name="img00_block1.pdf", feat="vgg1")
sw.draw_kneighbours(dm.get_img_index("img_00000000.jpg"), graph_name="img00_block2.pdf", feat="vgg2")