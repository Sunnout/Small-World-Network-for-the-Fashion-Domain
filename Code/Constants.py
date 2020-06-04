# Directory where we save the output files
FILES_DIR = "./Files/"

# Directory where we save the images created
RESULTS_DIR = "./Results/"

# Directory of the image database
DATA_DIR = "./Data/"

# Size of the sample set use to obtain the normalization values
SAMPLE_SET_SIZE = 10

# Standard name use for the column when writing files
# WARNING: When changing this make sure to change the np.savez and np.load statements as well
STD_COLUMN = "features"

NPZ_EXTENSION = ".npz"

# Files used for feature storage
HOC_MATRIX_FILE = "hoc_matrix"
HOG_MATRIX_FILE = "hog_matrix"
VGG_BLOCK1_MATRIX_FILE = "vgg16_block1_matrix"
VGG_BLOCK2_MATRIX_FILE = "vgg16_block2_matrix"
VGG_BLOCK3_MATRIX_FILE = "vgg16_block3_matrix"
VGG_BLOCK4_MATRIX_FILE = "vgg16_block4_matrix"
VGG_BLOCK5_MATRIX_FILE = "vgg16_block5_matrix"

FINAL_DISTANCES_FILE = "final_dist_matrix"
COLOR_NEIGH_FILE = "color_neighbours"
GRADS_NEIGH_FILE = "grads_neighbours"
VGG_BLOCK1_NEIGH_FILE = "vgg16_block1_neighbours"
VGG_BLOCK2_NEIGH_FILE = "vgg16_block2_neighbours"
VGG_BLOCK3_NEIGH_FILE = "vgg16_block3_neighbours"
VGG_BLOCK4_NEIGH_FILE = "vgg16_block4_neighbours"
VGG_BLOCK5_NEIGH_FILE = "vgg16_block5_neighbours"

KNN = "knn"

# Number of neighbours to extract
N_NEIGHBOURS = 10

# VGG16 Available layers

VGG16_BLOCK1_POOL_LAYER = "block1_pool"
VGG16_BLOCK2_POOL_LAYER = "block2_pool"
VGG16_BLOCK3_POOL_LAYER = "block3_pool"
VGG16_BLOCK4_POOL_LAYER = "block4_pool"
VGG16_BLOCK5_POOL_LAYER = "block5_pool"

# Dictionaries used to relate layers to their respective files

VGG_MATRIX_FILES = {
    VGG16_BLOCK1_POOL_LAYER : VGG_BLOCK1_MATRIX_FILE,
    VGG16_BLOCK2_POOL_LAYER : VGG_BLOCK2_MATRIX_FILE,
    VGG16_BLOCK3_POOL_LAYER : VGG_BLOCK3_MATRIX_FILE,
    VGG16_BLOCK4_POOL_LAYER : VGG_BLOCK4_MATRIX_FILE,
    VGG16_BLOCK5_POOL_LAYER : VGG_BLOCK5_MATRIX_FILE,
}

VGG_NEIGH_FILES = {
    VGG16_BLOCK1_POOL_LAYER : VGG_BLOCK1_NEIGH_FILE,
    VGG16_BLOCK2_POOL_LAYER : VGG_BLOCK2_NEIGH_FILE,
    VGG16_BLOCK3_POOL_LAYER : VGG_BLOCK3_NEIGH_FILE,
    VGG16_BLOCK4_POOL_LAYER : VGG_BLOCK4_NEIGH_FILE,
    VGG16_BLOCK5_POOL_LAYER : VGG_BLOCK5_NEIGH_FILE,
}

