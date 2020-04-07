from ._sparse import _rank, read_file, read_file_safe, _topk
from scipy.sparse import csr_matrix, isspmatrix
import numpy as np
import warnings
from sklearn.preprocessing import normalize as sk_normalize
from scipy.special import expit


def rank(X):
    '''Rank of each element in decreasing order (per-row)
    Ranking will start from one (with zero at zero entries)
    '''
    ranks = _rank(X.data, X.indices, X.indptr)
    return csr_matrix((ranks, X.indices, X.indptr), shape=X.shape)


def topk(X, k, pad_ind, pad_val, return_values=False, dtype='float32'):
    """Get top-k indices and values for a sparse (csr) matrix
    Arguments:
    ---------
    X: csr_matrix
        sparse matrix
    k: int
        values to select
    pad_ind: int
        padding index for indices array
        Useful when number of values in a row are less than k
    pad_val: int
        padding index for values array
        Useful when number of values in a row are less than k
    return_values: boolean, optional, default=False
        Return topk values or not
    
    Returns:
    --------
    ind: np.ndarray
        topk indices; size=(num_rows, k)
    val: np.ndarray, optional
        topk val; size=(num_rows, k)
    """
    ind, val = _topk(X.data, X.indices, X.indptr, k, pad_ind, pad_val)
    if return_values:
        return ind, val.astype(dtype)
    else:
        return ind


def retain_topk(X, copy=True, k=5):
    """Retain topk values of each row and make everything else zero
    Arguments:
    ---------
    X: csr_matrix
        sparse matrix
    copy: boolean, optional, default=True
        copy data or change original array
    k: int, optional, default=5
        retain these many values
   
    Returns:
    --------
    X: csr_matrix
        sparse mat with only k entries in each row
    """
    ranks = rank(X)
    if copy:
        X = X.copy()
    X[ranks > k] = 0.0
    X.eliminate_zeros()
    return X


def binarize(X, copy=False):
    """Binarize a sparse matrix
    """
    if copy:
        X = X.copy()
    X.data.fill(1)
    return X


def gen_shape(indices, indptr, zero_based=True):
    _min = min(indices)
    if not zero_based:
        indices = list(map(lambda x: x-_min, indices))
    num_cols = max(indices)
    num_rows = len(indptr) - 1
    return (num_rows, num_cols)


def expand_indptr(num_rows_inferred, num_rows, indptr):
    """Expand indptr if inferred num_rows is less than given
    """
    _diff = num_rows - num_rows_inferred
    if _diff > 0:  # Fix indptr as per new shape
        # Data is copied here
        warnings.warn("Header mis-match from inferred shape!")
        return np.concatenate((indptr, np.repeat(indptr[-1], _diff)))
    elif _diff == 0:  # It's fine
        return indptr
    else:
        raise NotImplementedError("Unknown behaviour!")


def tuples_to_sparse(X, shape=None, dtype='float32', zero_based=True):
    """Convert a list of list of tuples to a csr_matrix
    Arguments:
    ---------
    X: list of list of tuples
        nnz indices for each row
    shape: tuple or none, optional, default=None
        Use this shape or infer from data
    dtype: 'str', optional, default='float32'
        datatype for data
    zero_based: boolean or "auto", default=True
        indices are zero based or not
  
    Returns:
    --------
    X: csr_matrix
    """
    indices = []
    data = []
    indptr = [0]
    offset = 0
    for item in X:
        if len(item) > 0:
            indices += list(map(lambda x: x[0], item))
            data += list(map(lambda x: x[1], item))
            offset += len(item)
        indptr.append(offset)
    _shape = gen_shape(indices, indptr, zero_based)
    if shape is not None:
        assert _shape[0] <= shape[0], "num_rows_inferred > num_rows_given"
        assert _shape[1] <= shape[1], "num_cols_inferred > num_cols_given"
        indptr = expand_indptr(_shape[0], shape[0], indptr)
    return csr_matrix(
        (np.array(data, dtype=dtype), np.array(indices), np.array(indptr)),
        shape=shape)


def ll_to_sparse(X, shape=None, dtype='float32', zero_based=True):
    """Convert a list of list to a csr_matrix; All values are 1.0
    Arguments:
    ---------
    X: list of list of tuples
        nnz indices for each row
    shape: tuple or none, optional, default=None
        Use this shape or infer from data
    dtype: 'str', optional, default='float32'
        datatype for data
    zero_based: boolean or "auto", default=True
        indices are zero based or not
   
    Returns:
    -------
    X: csr_matrix
    """
    indices = []
    indptr = [0]
    offset = 0
    for item in X:
        if len(item) > 0:
            indices.extend(item)
            offset += len(item)
        indptr.append(offset)
    data = [1.0]*len(indices)
    _shape = gen_shape(indices, indptr, zero_based)
    if shape is not None:
        assert _shape[0] <= shape[0], "num_rows_inferred > num_rows_given"
        assert _shape[1] <= shape[1], "num_cols_inferred > num_cols_given"
        indptr = expand_indptr(_shape[0], shape[0], indptr)
    return csr_matrix(
        (np.array(data, dtype=dtype), np.array(indices), np.array(indptr)),
        shape=shape)


def normalize(X, norm='l2', copy=False):
    """Normalize sparse or dense matrix
    Arguments:
    ---------
    X: csr_matrix or csc_matrix
        sparse matrix
    norm: str, optional, default='l2'
        normalize with l1/l2
    copy: boolean, optional, default=False 
        whether to copy data or not
    """
    features = sk_normalize(X, norm=norm, copy=copy)
    return features


def _read_file_safe(f, dtype, zero_based, query_id,
                    offset=0, length=-1, header=True):
    def _handle_header(f, header):
        num_cols, num_rows = None, None
        if header:
            num_cols, num_rows = map(
                int, f.readline().decode('utf-8').strip().split(' '))
        return f, (num_cols, num_rows)
    if hasattr(f, "read"):
        f, _header_shape = _handle_header(f, header)
        actual_dtype, data, ind, indptr, query = \
            read_file_safe(f, dtype, zero_based, query_id,
                           offset, length)
    else:
        with _gen_open(f) as f:
            f, _header_shape = _handle_header(f, header)
            actual_dtype, data, ind, indptr, query = \
                read_file_safe(f, dtype, zero_based, query_id,
                               offset, length)

    data = np.frombuffer(data, actual_dtype)
    indices = np.frombuffer(ind, np.int64)
    indptr = np.frombuffer(indptr, dtype=np.int64)   # never empty
    query = np.frombuffer(query, np.int64)
    data = np.asarray(data, dtype=dtype)    # no-op for float{32,64}
    return data, indices, indptr, query, _header_shape


def _read_file(f, dtype, zero_based, query_id,
               offset=0, length=-1, header=True):
    def _handle_header(f, header):
        num_cols, num_rows = None, None
        if header:
            num_cols, num_rows = map(
                int, f.readline().decode('utf-8').strip().split(' '))
        return f, (num_cols, num_rows)
    if hasattr(f, "read"):
        f, _header_shape = _handle_header(f, header)
        actual_dtype, data, rows, cols, query = \
            read_file(f, dtype, zero_based, query_id,
                      offset, length)
    else:
        with _gen_open(f) as f:
            f, _header_shape = _handle_header(f, header)
            actual_dtype, data, rows, cols, query = \
                read_file(f, dtype, zero_based, query_id,
                          offset, length)

    data = np.frombuffer(data, actual_dtype)
    rows = np.frombuffer(rows, np.int64)
    cols = np.frombuffer(cols, dtype=np.int64)   # never empty
    query = np.frombuffer(query, np.int64)
    data = np.asarray(data, dtype=dtype)    # no-op for float{32,64}
    return data, rows, cols, query, _header_shape


def _gen_open(f, _mode='rb'):
    # Update this for more generic file types
    return open(f, _mode)


def sigmoid(X):
    """Sparse sigmoid i.e. zeros are kept intact
    Parameters
    ----------
    X: csr_matrix
        sparse matrix in csr format
    """
    X.data = expit(X.data)
    return X


def _map_rows(X, mapping, shape):
    """Indices should not be repeated
    Will not convert to dense
    """
    X = X.tocsr()  # Avoid this?
    row_idx, col_idx = X.nonzero()
    vals = np.array(X[row_idx, col_idx]).squeeze()
    row_indices = list(map(lambda x: mapping[x], row_idx))
    return csr_matrix(
        (vals, (np.array(row_indices), np.array(col_idx))), shape=shape)


def _map_cols(X, mapping, shape):
    """Indices should not be repeated
    Will not convert to dense
    """
    X = X.tocsr()
    row_idx, col_idx = X.nonzero()
    vals = np.array(X[row_idx, col_idx]).squeeze()
    col_indices = list(map(lambda x: mapping[x], col_idx))
    return csr_matrix(
        (vals, (np.array(row_idx), np.array(col_indices))), shape=shape)


def _map(X, mapping, shape, axis=1):
    """Map sparse matrix as per given mapping
    """
    if axis == 1:
        return _map_cols(X, mapping, shape)
    elif axis == 0:
        return _map_rows(X, mapping, shape)
    else:
        raise NotImplementedError("Unknown axis for sparse matrix!")
