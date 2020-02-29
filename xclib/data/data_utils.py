"""
Data tools for multi-labels datasets
Uses sparse matrices which is suitable for large datasets
"""
import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import operator
from ..utils.sparse import ll_to_sparse, expand_indptr, _read_file, _read_file_safe
import six
import warnings
import json
import gzip


__author__ = 'X'


def read_split_file(split_fname):
    """Return array of train/ test split
    
    Arguments
    ----
    split_fname: str: split file with 0/1 in each line
    
    Returns
    -------
    np.array: split
    """
    return np.genfromtxt(split_fname, dtype=np.int)


def _split_data(data, indices):
    """Handles list and sparse/dense matrices
    
    Arguments
    -----
    data: list or matrix
        given data
    indices: indices to choose
        fetch these indices

    Returns
    -------
    chosen data in the same format
    """
    if isinstance(data, list):
        return list(operator.itemgetter(*indices)(data))
    else:
        return data[indices, :]


def split_train_test(features, labels, split):
    """Split a list of text in train and text set
    *0: train, 1: test
    
    Arguments
    ----
    features: list or dense or sparse matrix
    labels: list or dense or sparse matrix
    split: numpy array with 0/1 split
    
    Returns
    -------
    train_feat, train_labels
        train features and labels
    test_feat, test_labels
        test features and labels
    """
    train_idx = np.where(split == 0)[0].tolist()
    test_idx = np.where(split == 1)[0].tolist()
    train_feat, test_feat = _split_data(features, train_idx), _split_data(
        features, test_idx)
    train_labels, test_labels = _split_data(labels, train_idx), _split_data(
        labels, test_idx)
    return train_feat, train_labels, test_feat, test_labels


def write_sparse_file(X, filename, header=True):
    """Write sparse label matrix to text file (comma separated)
    Header: (#users, #labels)
    
    Arguments
    ----
    X: sparse matrix
        data to be written
    filename: str
        output file
    header: bool, default=True
        write header or not
    """
    if not isinstance(X, csr_matrix):
        X = X.tocsr()
    X.sort_indices()
    with open(filename, 'w') as f:
        if header:
            print("%d %d" % (X.shape[0], X.shape[1]), file=f)
        for y in X:
            idx = y.__dict__['indices']
            val = y.__dict__['data']
            sentence = ' '.join(['{}:{}'.format(x, v)
                                 for x, v in zip(idx, val)])
            print(sentence, file=f)


def read_sparse_file(file, n_features=None, dtype='float32', zero_based=True,
                     query_id=False, offset=0, length=-1, header=True,
                     force_header=True, safe_read=True):
    """Read sparse file as a scipy.sparse matrix

    Arguments
    ----
    file: str
        input file in libsvm format with or without header
    n_features: int or None, default=None
        number of features
    dtype: str, default='float32'
        data type of values
    zero_based: str or boolean, default=True
        zero based indices
    query_id: bool, default=False
        If True, will return the query_id array
    offset: int, default=0
        Ignore the offset first bytes by seeking forward, 
        then discarding the following bytes up until the next new line character.
    length: int, default=-1
        If strictly positive, stop reading any new line of data once the position 
        in the file has reached the (offset + length) bytes threshold.
    header: bool, default=True
        does file have a header
    force_header: bool, default=True
        force the shape of header
    safe_read: bool, default=True 
        check for sorted and unique indices and checks for header_shape and inferred shape
        use False when indices are not sorted

    Returns
    ------- 
    X: scipy.sparse.csr_matrix
        data in sparse format
    query_id: array of shape (n_samples,)
        query ids
    """
    if (offset != 0 or length > 0) and zero_based == "auto":
        zero_based = True

    if (offset != 0 or length > 0) and n_features is None:
        raise ValueError(
            "n_features is required when offset or length is specified.")

    if safe_read:
        data, indices, indptr, \
            query_values, _header_shape = _read_file_safe(file,
                                                          dtype,
                                                          bool(zero_based),
                                                          bool(query_id),
                                                          offset=offset,
                                                          length=length,
                                                          header=header)
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
    else:  # Just use header shape
        data, rows, cols, \
            query_values, _header_shape = _read_file(file,
                                                     dtype,
                                                     bool(zero_based),
                                                     bool(query_id),
                                                     offset=offset,
                                                     length=length,
                                                     header=header)
        if (zero_based is False or zero_based == "auto" and (len(cols) > 0 and np.min(cols) > 0)):
            cols -= 1
        # Will sum if indices are repeated
        X = csr_matrix((data, (rows, cols)), shape=_header_shape)
    X.sort_indices()
    if query_id:
        return tuple(X, query_values)
    else:
        return X


def write_data(filename, features, labels, header=True):
    """Write data in sparse format

    Arguments
    ---------
    filename: str
        output file name
    features: csr_matrix
        features matrix
    labels: csr_matix
        labels matrix
    """
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
    """Read data in sparse format

    Arguments
    ---------
    filename: str
        output file name
    header: bool, default=True
        If header is present or not
    dtype: str, default='float32'
        data type of values
    zero_based: boolean, default=True
        zwero based indices?

    Returns
    --------
    features: csr_matrix
        features matrix
    labels: csr_matix
        labels matrix
    num_samples: int
        #instances
    num_feat: int
        #features
    num_labels: int
        #labels
    """
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
        labels = ll_to_sparse(
            labels, dtype=dtype, zero_based=zero_based, shape=_l_shape)
    return features, labels, num_samples, num_feat, num_labels


def read_corpus(fname):
    """
    Read gzip file with one json string in each line

    Returns
    -------
    A generator with dictionary as values
    """
    with gzip.open(fname, 'r') as fp:
        for line in fp:
            yield json.loads(line)


def write_corpus(fname, uid, title, text, label):
    """
    Write gzip file with one json string in each line

    Arguments
    ---------
    fname: str
        path of the output file
    uid: list of str/int
        unique identifier for each document
    title: list of str
        title of each document
    text: list of str
        text/body of each document
    title: list of str
        title of each document
    label: scipy.sparse matrix or list of list
        * will convert of list of list if sparse matrix is provided
        * 1.0 if relevance score is not available
    """
    def _sanity_check(*args):
        return all(len(x) == len(args[0]) for x in args)

    def _create_json_str(_uid, _title, _text, _label_ind, label_rel):
        return json.dumps(
            {'uid': _uid, 'title': _title,
             'content': _text, 'target_ind': _label_ind,
             'target_rel': label_rel})

    if issparse(label):
        label.eliminate_zeros()
        label_ind = label.tolil().rows  # list of list
        label_rel = label.tolil().data  # list of list
    else:
        label_ind = label
        label_rel = []
        for item in label_ind:
            label_rel.append([1.0]*len(item))

    assert _sanity_check(uid, title, text, label_ind,
                         label_rel), "#documents must be same in each object!"

    with gzip.open(fname, 'wt') as fp:
        for item in zip(uid, title, text, label_ind, label_rel):
            fp.write(_create_json_str(*item) + '\n')
