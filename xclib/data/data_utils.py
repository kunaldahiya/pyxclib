"""
        Data tools for multi-labels datasets
        Uses sparse matrices which is suitable for large datasets
"""
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import operator
from ..utils.sparse import ll_to_sparse, expand_indptr, _read_file
import six
import warnings

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


def write_sparse_file(X, filename, header=True):
    '''
        Write sparse label matrix to text file (comma separated)
        Header: (#users, #labels)
        Args:
            X: sparse matrix: labels
            filename: str: output file
            header: bool: include header or not
    '''
    if not isinstance(X, csr_matrix):
        X = X.tocsr()
    with open(filename, 'w') as f:
        if header:
            print("%d %d" % (X.shape[0], X.shape[1]), file=f)
        for y in X:
            idx = y.__dict__['indices']
            val = y.__dict__['data']
            sentence = ' '.join(['{}:{}'.format(x, v)
                                 for x, v in zip(idx, val)])
            print(sentence, file=f)


def read_sparse_file(file, n_features=None, dtype=np.float64, 
                     zero_based=True, query_id=False, offset=0, 
                     length=-1, header=True, force_header=True):
    '''
        Args:
            file: str: input file in libsvm format with or without header
            n_features: int/None: number of features
            dtype:: for values
            zero_based: str or boolean: zero based indices
            query_id: bool: If True, will return the query_id array
            offset: int: Ignore the offset first bytes by seeking forward,
                        then discarding the following bytes up until the
                        next new line character.
            lenght: int: If strictly positive, stop reading any new line
                        of data once the position in the file has reached
                        the (offset + length) bytes threshold.
            header: bool: does file have a header
            force_header: bool: force the shape of header
        Returns:
            X: scipy.sparse.csr_matrix
            query_id: array of shape (n_samples,)
    '''
    if (offset != 0 or length > 0) and zero_based == "auto":
        zero_based = True

    if (offset != 0 or length > 0) and n_features is None:
        raise ValueError(
            "n_features is required when offset or length is specified.")

    data, indices, indptr, \
        query_values, _header_shape = _read_file(file,
                                                 dtype,
                                                 bool(zero_based),
                                                 bool(query_id),
                                                 offset=offset,
                                                 length=length)

    if (zero_based is False or zero_based == "auto" and (len(indices) > 0 and np.min(indices) > 0)):
        indices -= 1
    n_f = (indices.max() if len(indices) else 0) + 1  # Num features
    if n_features is None:
        n_features = n_f
    elif n_features < n_f:
        raise ValueError("n_features was set to {},"
                         " but input file contains {} features"
                         .format(n_features, n_f))

    shape = (indptr.shape[0] - 1, n_features)
    # Throw warning if shapes do not match
    if header and shape != _header_shape:
        warnings.warn("Header mis-match from inferred shape!")
    if header and force_header:
        # Inferred shape must be lower than header in both dimensions
        assert shape[0] <= _header_shape[0], "num_rows_inferred > num_rows_header"
        assert shape[1] <= _header_shape[1], "num_cols_inferred > num_cols_header"
        indptr = expand_indptr(shape[0], _header_shape[0], indptr)
        shape = _header_shape
    X = csr_matrix((data, indices, indptr), shape)
    X.sort_indices()
    if query_id:
        return tuple(X, query_values)
    else:
        return X


def write_data(filename, features, labels, header=True):
    '''
        Write data in sparse format
        Args:
            filename: str: output file name
            features: csr_matrix: features matrix
            labels: csr_matix: labels matrix

    '''
    if header:
        with open(filename, 'w') as f:
            out = "{} {} {}".format(
                features.shape[0], features.shape[1], labels.shape[1])
            print(out, file=f)
        with open(filename, 'ab') as f:
            dump_svmlight_file(features, labels, f, multilabel=True)
    else:
        with open(filename, 'wb') as f:
            dump_svmlight_file(features, labels, f, multilabel=True)


def read_data(filename, header=True, dtype='float32', zero_based=True):
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
        _l_shape = None
        if header:
            line = f.readline().decode('utf-8').rstrip("\n")
            line = line.split(" ")
            num_samples, num_feat, num_labels = int(
                line[0]), int(line[1]), int(line[2])
            _l_shape = (num_samples, num_labels)
        else:
            num_samples, num_feat, num_labels = None, None, None
        features, labels = load_svmlight_file(f, multilabel=True)
        labels = ll_to_sparse(labels, dtype=dtype, zero_based=zero_based, shape=_l_shape)
    return features, labels, num_samples, num_feat, num_labels
