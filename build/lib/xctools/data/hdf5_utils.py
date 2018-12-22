"""
        Read multi-label data and write into HDF5 format
"""
import h5sparse
import sys
import data_utils
import numpy as np
from scipy.sparse import csr_matrix

__author__ = 'KD'

def read_data(fname):
    """
        Read the h5py file and return features, labels and attributes 
    """
    h5f = h5sparse.File(fname)
    temp = h5f['data/attributes'][:].toarray().tolist()[0]
    num_samples, num_feat, num_labels = temp[0], temp[1], temp[2]
    features = h5f['data/features']
    labels = h5f['data/labels']
    return features, labels, num_samples, num_feat, num_labels


def write_data(in_file, out_file):
    """
        Read data from input file and write to sparse hdf5 file
        Format: 
        feature matrix: 'data/features'
        label matrix: 'data/labels'
    """
    features, labels, num_samples, num_feat, num_labels = data_utils.read_data(
        in_file)
    attributes = np.array(
        [num_samples, num_feat, num_labels], dtype=np.int).reshape(1, 3)
    attributes = csr_matrix(attributes)
    labels = data_utils.binarize_labels(labels, num_labels)
    with h5sparse.File(out_file) as h5f:
        h5f.create_dataset('data/attributes', data=attributes)
        h5f.create_dataset('data/features', data=features)
        h5f.create_dataset('data/labels', data=labels)


if __name__ == '__main__':
    create(sys.argv[1], sys.argv[2])
