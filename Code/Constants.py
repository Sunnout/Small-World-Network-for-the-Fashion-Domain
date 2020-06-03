# Directory where we save the output files
FILES_DIR = "Files/"

# Directory where we save the images created
RESULTS_DIR = "Results/"

# Directory of the image database
DATA_DIR = "Data/"

# Size of the sample set use to obtain the normalization values
SAMPLE_SET_SIZE = 10

# Standard name use for the column when writing files
# WARNING: When changing this make sure to change the np.savez statements as well
STD_COLUMN = "features"

# Files used for feature storage
HOC_MATRIX_FILE = "hoc_matrix.npz"
HOG_MATRIX_FILE = "hog_matrix.npz"
VGG_BLOCK1_MATRIX_FILE = "vgg16_block1_matrix.npz"
VGG_BLOCK2_MATRIX_FILE = "vgg16_block2_matrix.npz"
VGG_BLOCK3_MATRIX_FILE = "vgg16_block3_matrix.npz"

FINAL_DISTANCES_FILE = "final_dist_matrix.npz"
COLOR_NEIGH_FILE = "color_neighbours.npz"
GRADS_NEIGH_FILE = "grads_neighbours.npz"
VGG_BLOCK1_NEIGH_FILE = "vgg16_block1_neighbours.npz"
VGG_BLOCK2_NEIGH_FILE = "vgg16_block2_neighbours.npz"
VGG_BLOCK3_NEIGH_FILE = "vgg16_block3_neighbours.npz"

KNN = "knn"


