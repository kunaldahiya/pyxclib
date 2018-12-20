"""
        Data tools for multi-labels datasets
        Uses sparse matrices which is suitable for large datasets
"""
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import operator

__author__ = 'KD'

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
    with open(filename, 'w') as f:
        if header:
            f.write(str(labels.shape[0])+" "+str(labels.shape[1])+"\n")
        non_zero_rows, non_zero_cols = labels.nonzero()
        temp=''
        for i in range(len(non_zero_rows)):
            try:
                is_last_element = non_zero_rows[i] != non_zero_rows[i+1]
            except IndexError:
                is_last_element = True
            temp = temp + str(non_zero_cols[i])+':'+str(labels[non_zero_rows[i], non_zero_cols[i]])+" "
            if is_last_element:
                f.write(temp.rstrip(" ")+"\n")
                temp=''

def read_sparse_file(filename, header=True):
    '''
        Args:
            input file in libsvm format
        Returns: 
            CSR matrix
    '''
    with open(filename, 'r') as f:
        if header:
            num_samples, num_labels = f.readline().rstrip('\n').split(' ')
            num_samples, num_labels = int(num_samples), int(num_labels)
        else:
            NotImplementedError("Not yet implemented!")
        data = lil_matrix((num_samples, num_labels), dtype=np.float32)
        line_num = 0
        for line in f:
            temp = line.split(' ')
            for item in temp:
                item2 = item.split(':')
                try:
                    data[line_num, int(item2[0])] = float(item2[1])
                except IndexError:
                    pass
                #print("Column error in row: ", line_num)
            line_num+=1
    return data.tocsr()

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
