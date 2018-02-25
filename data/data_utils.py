"""
        Data tools for multi-labels datasets
        Uses sparse matrices which is suitable for large datasets
"""
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.datasets import load_svmlight_file, dump_svmlight_file

__author__='KD'

def read_split_file(split_fname):
    '''
        Return array of train/ test split
    '''
    return np.genfromtxt(split_fname, dtype=np.int)

def train_test_split(features, labels, split_fname):
    '''
        Input: features and labels
        Returns: Train and test split based on (0:train and 1:test)
    '''
    split = read_split_file(split_fname)
    train_features = features[split==0, :]
    test_features = features[split==1, :]
    train_labels = labels[split==0, :]
    test_labels = labels[split==1, :]
    return train_features, train_labels, test_features, test_labels

def write_sparse_file(labels, filename, header=True):
    '''
        Write sparse label matrix to text file (comma separated)
        Header: (#users, #labels)
    '''
    with open(filename, 'w') as f:
        f.write(str(labels.shape[0])+" "+str(labels.shape[1])+"\n")
        non_zero_rows, non_zero_cols = labels.nonzero()
        temp=''
        for i in range(len(non_zero_rows)):
            try:
                is_last_element = non_zero_rows[i]!=non_zero_rows[i+1]
            except IndexError:
                is_last_element = True
            temp = temp + str(non_zero_cols[i])+':'+str(labels[non_zero_rows[i], non_zero_cols[i]])+" "
            if is_last_element:
                f.write(temp.rstrip(" ")+"\n")
                temp=''

def read_sparse_file(filename, header=True):
    '''
        Input: Sparse file (idx:value)
        Returns: CSR matrix
    '''
    with open(filename, 'r') as f:
        if header:
            num_samples, num_labels = f.readline().rstrip('\n').split(' ')
            num_samples, num_labels = int(num_samples), int(num_labels)
        else:
            NotImplementedError("Not yet implemented!")
        data = lil_matrix((num_samples, num_labels), dtype=np.float)
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
        Write data in sparse format (without header, write manually if needed)
        labels idx:value
    '''
    with open(filename, 'wb') as f:
        dump_svmlight_file(features, labels, f, multilabel=True)

def read_data(filename, header=True):
    '''
        Read data in sparse format
        Returns: features and label matrix (in sparse format), #samples, #features, #labels
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
        Input: list of list
        Returns: csr_matrix with positive labels as 1
    '''
    temp = lil_matrix((len(labels), num_classes), dtype=np.int)
    for i in range(len(labels)):
        for item in labels[i]:
            temp[i, int(item)] = 1
    return temp.tocsr()
