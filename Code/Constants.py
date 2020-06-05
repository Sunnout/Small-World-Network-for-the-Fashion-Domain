# Directory of the image database
DATA_DIR = "./Data/"

# Directory where the output files are saved
FILES_DIR = "./Files/"

# Directory the images created are saved
RESULTS_DIR = "./Results/"

# Extension of the storage files
NPZ_EXTENSION = ".npz"

# Standard name used for the column when writing the feature matrices in npz files
# WARNING: When changing this constant, make sure to change the column name in np.savez statements!
FEATURE = "features"

# Files used for feature storage
HOC_MATRIX_FILE = "hoc_matrix"
HOG_MATRIX_FILE = "hog_matrix"
VGG_BLOCK1_MATRIX_FILE = "vgg16_block1_matrix"
VGG_BLOCK2_MATRIX_FILE = "vgg16_block2_matrix"
VGG_BLOCK3_MATRIX_FILE = "vgg16_block3_matrix"
VGG_BLOCK4_MATRIX_FILE = "vgg16_block4_matrix"
VGG_BLOCK5_MATRIX_FILE = "vgg16_block5_matrix"

# Standard name used for the column when writing the neighbours in npz files
# WARNING: When changing this constant, make sure to change the column name in np.savez statements!
KNN = "knn"

# Files used for k-NN storage according to a single feature
COLOR_NEIGH_FILE = "color_neighbours"
GRADS_NEIGH_FILE = "grads_neighbours"
VGG_BLOCK1_NEIGH_FILE = "vgg16_block1_neighbours"
VGG_BLOCK2_NEIGH_FILE = "vgg16_block2_neighbours"
VGG_BLOCK3_NEIGH_FILE = "vgg16_block3_neighbours"
VGG_BLOCK4_NEIGH_FILE = "vgg16_block4_neighbours"
VGG_BLOCK5_NEIGH_FILE = "vgg16_block5_neighbours"

# Standard name used for the column when writing the final distances in the npz file
# WARNING: When changing this constant, make sure to change the column name in np.savez statement!
DIST = "dist"

# File used for final distance storage
FINAL_DISTANCES_FILE = "final_dist_matrix"

# Standard name used for the column when writing the image names in the npz file
# WARNING: When changing this constant, make sure to change the column name in np.savez statement!
IMG_NAMES = "names"

# Size of the sample set used to obtain the normalization values
SAMPLE_SET_SIZE = 10

# Number of neighbours to extract for each feature
N_NEIGHBOURS = 10

# Number of edges to add for each node in the graph
EDGES = 5

# VGG16 Available layers
VGG16_BLOCK1_POOL_LAYER = "block1_pool"
VGG16_BLOCK2_POOL_LAYER = "block2_pool"
VGG16_BLOCK3_POOL_LAYER = "block3_pool"
VGG16_BLOCK4_POOL_LAYER = "block4_pool"
VGG16_BLOCK5_POOL_LAYER = "block5_pool"

# Dictionary used to relate VGG16 layers to their respective feature files
VGG_MATRIX_FILES = {
    VGG16_BLOCK1_POOL_LAYER: VGG_BLOCK1_MATRIX_FILE,
    VGG16_BLOCK2_POOL_LAYER: VGG_BLOCK2_MATRIX_FILE,
    VGG16_BLOCK3_POOL_LAYER: VGG_BLOCK3_MATRIX_FILE,
    VGG16_BLOCK4_POOL_LAYER: VGG_BLOCK4_MATRIX_FILE,
    VGG16_BLOCK5_POOL_LAYER: VGG_BLOCK5_MATRIX_FILE,
}

# Dictionary used to relate VGG16 layers to their respective neighbour files
VGG_NEIGH_FILES = {
    VGG16_BLOCK1_POOL_LAYER: VGG_BLOCK1_NEIGH_FILE,
    VGG16_BLOCK2_POOL_LAYER: VGG_BLOCK2_NEIGH_FILE,
    VGG16_BLOCK3_POOL_LAYER: VGG_BLOCK3_NEIGH_FILE,
    VGG16_BLOCK4_POOL_LAYER: VGG_BLOCK4_NEIGH_FILE,
    VGG16_BLOCK5_POOL_LAYER: VGG_BLOCK5_NEIGH_FILE,
}
