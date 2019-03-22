from scipy.io import loadmat
from xctools.data import data_utils
import sys

mat_file=loadmat(sys.argv[1])
data_utils.write_sparse_file(mat_file['predicted_labels'],sys.argv[2],header=True)