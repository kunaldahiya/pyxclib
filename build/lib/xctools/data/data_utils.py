"""
        Data tools for multi-labels datasets
        Uses sparse matrices which is suitable for large datasets
"""
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.preprocessing import normalize
import operator

__author__ = 'X'

def read_split_file(split_fname):
    '''
        Return array of train/ test split
        Args:
            split_fname: str: split file with 0/1 in each line
        Returns:
            np.array: split
    '''
    return np.genfromtxt(split_fname, dtype=np.int)

def _split_data(data, indices):
    """
        Handles list and sparse/dense matrices
        Args:
            data: list of matrix
            indices: indices to choose
        Returns:
            chosen data in the same format
    """
    if isinstance(data, list):
        return list(operator.itemgetter(*indices)(data))
    else:
        return data[indices, :]


def split_train_test(features, labels, split):
    '''
        Split a list of text in train and text set
        0: train, 1: test
        Args:
            features: list or dense or sparse matrix
            labels: list or dense or sparse matrix
            split: numpy array with 0/1 split
        Returns:
            train_feat, train_labels: train features and labels
            test_feat, test_labels: test features and labels
    '''
    train_idx = np.where(split == 0)[0].tolist()
    test_idx = np.where(split == 1)[0].tolist()
    train_feat, test_feat = _split_data(features, train_idx), _split_data(
        features, test_idx)
    train_labels, test_labels = _split_data(labels, train_idx), _split_data(
        labels, test_idx)
    return train_feat, train_labels, test_feat, test_labels


def write_sparse_file(labels, filename, header=True):
    '''
        Write sparse label matrix to text file (comma separated)
        Header: (#users, #labels)
        Args:
            labels: sparse matrix: labels
            filename: str: output file
            header: bool: include header or not
    '''
    if not isinstance(labels, csr_matrix):
        labels=labels.tocsr()
    with open(filename, 'w') as f:
        if header:
            print("%d %d"%(labels.shape[0],labels.shape[1]),file=f)
        for y in labels:
            idx = y.__dict__['indices']
            val = y.__dict__['data']
            sentence = ' '.join(['{}:{}'.format(x,v) for x,v in zip(idx,val)])
            print(sentence,file=f)


def read_sparse_file(filename, header=True):
    '''
        Args:
            input file in libsvm format
        Returns: 
            CSR matrix
    '''
    with open(filename, 'r') as f:
        if header:
            num_samples, num_labels = map(int,f.readline().strip().split(' '))
        else:
            NotImplementedError("Not yet implemented!")
        data = lil_matrix((num_samples, num_labels), dtype=np.float64)
        for i,line in enumerate(f):
            temp = [x for x in list(map(lambda x:x.split(':') ,line.strip().split(' '))) if x[0] !='']
            if len(temp)>0:
                idx = np.asarray(list(map(lambda x: np.int64(x[0]),temp)))
                val = np.asarray(list(map(lambda x: np.float64(x[1]),temp)))
                data[i, idx] = val
    return data.tocsr()

            
# def read_sparse_file(filename, header=True):
#     '''
#         Args:
#             input file in libsvm format
#         Returns: 
#             CSR matrix
#     '''
#     with open(filename, 'r') as f:
#         if header:
#             num_samples, num_labels = map(int,f.readline().strip().split(' '))
#         else:
#             NotImplementedError("Not yet implemented!")
#         data = lil_matrix((num_samples, num_labels), dtype=np.float32)
#         for i,line in enumerate(f):
#             temp = [x for x in list(map(lambda x:x.split(':') ,line.strip().split(' '))) if x[0] !='']
#             if len(temp)>0:
#                 idx = np.asarray(list(map(lambda x: np.int32(x[0]),temp)))
#                 val = np.asarray(list(map(lambda x: np.float32(x[1]),temp)))
#                 data[i, idx] = val
#     return data.tocsr()

def write_data(filename, features, labels):
    '''
        Write data in sparse format
        Args:
            filename: str: output file name
            features: csr_matrix: features matrix
            labels: csr_matix: labels matrix

    '''
    with open(filename, 'wb') as f:
        dump_svmlight_file(features, labels, f, multilabel=True)

def read_data(filename, header=True):
    '''
        Read data in sparse format
        Args:
            filename: str: output file name
            header: bool: If header is present or not
        Returns:
            features: csr_matrix: features matrix
            labels: csr_matix: labels matrix
            num_samples: int: #instances
            num_feat: int: #features
            num_labels: int: #labels

    '''
    with open(filename, 'rb') as f:
        if header:
            line = f.readline().decode('utf-8').rstrip("\n")
            line = line.split(" ")
            num_samples, num_feat, num_labels = int(line[0]), int(line[1]), int(line[2])
        else:
            num_samples, num_feat, num_labels = None, None, None
        features, labels = load_svmlight_file(f, multilabel=True)
    return features, labels, num_samples, num_feat, num_labels

def binarize_labels(labels, num_classes):
    '''
        Binarize labels
        Args:
            labels: list of list
        Returns: 
            csr_matrix with positive labels as 1
    '''
    temp = lil_matrix((len(labels), num_classes), dtype=np.int)
    for idx, _lb in enumerate(labels):
        for item in _lb:
            temp[idx, int(item)] = 1
    return temp.tocsr()


def tuples_to_csr(_input, _shape):
    """
        Convert a list of list of tuples to csr matrix
        Args:
        _input: list
        _shape: tuple: shape of output matrix
        Returns:
        output: csr_matrix: matrix with given data and shape
    """
    rows = []
    cols = []
    vals = []
    for idx, item in enumerate(_input):
        if len(item)>0:
            row+=[idx]*len(item)
            cols+=list(map(lambda x: x[0],item))
            vals+=list(map(lambda x: x[1],item))
    return csr_matrix(np.array(vals), (np.array(rows), np.array(cols)), shape=_shape)

def normalize_data(features, norm='l2'):
    features = normalize(features, norm='l2', copy=True)
    return features 