from Code.SmallWorldGraph import SmallWorldGraph
from Code.DataManager import DataManager as dm
from Code.Constants import VGG16_BLOCK2_POOL_LAYER, VGG16_BLOCK3_POOL_LAYER, VGG16_BLOCK4_POOL_LAYER

print("Imports Finished!")

sw = SmallWorldGraph(update=False, compute_final_distances=False,
                     layers=[VGG16_BLOCK2_POOL_LAYER, VGG16_BLOCK3_POOL_LAYER, VGG16_BLOCK4_POOL_LAYER])

print("Created graph!")

sw.show_shortest_path(dm.get_img_index("img_00000001.jpg"), dm.get_img_index("img_00000052.jpg"),
                      img_size=0.1, graph_name="1_52_path.pdf")