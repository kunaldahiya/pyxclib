from ._sparse import _rank, read_file, read_file_safe, _topk
from scipy.sparse import csr_matrix, isspmatrix, coo_matrix
import numpy as np
import warnings
from sklearn.preprocessing import normalize as sk_normalize
from scipy.special import expit
import numba as nb


def binarize(X, copy=False):
    """Binarize a sparse matrix
    """
    if copy:
        X = X.copy()
    X.data.fill(1)
    return X


def frequency(X, axis=0, copy=False):
    '''Count non-zeros 
    Arguments:
    ----------
    X: csr_matrix
        sparse matrix to process
    axis: int, optional, default=0
        reduce along this axis
    '''
    X = binarize(X, copy)
    return np.ravel(X.sum(axis=axis))


def rank(X):
    '''Rank of each element in decreasing order (per-row)
    Ranking will start from one (with zero at zero entries)
    '''
    ranks = _rank(X.data, X.indices, X.indptr)
    return csr_matrix((ranks, X.indices, X.indptr), shape=X.shape)


@nb.njit(parallel=True)
def _topk_nb(data, indices, indptr, k, pad_ind, pad_val):
    """Get top-k indices and values for a sparse (csr) matrix
    * Parallel version: uses numba
    Arguments:
    ---------
    data: np.ndarray
        data / vals of csr array
    indices: np.ndarray
        indices of csr array
    indptr: np.ndarray
        indptr of csr array
    k: int
        values to select
    pad_ind: int
        padding index for indices array
        Useful when number of values in a row are less than k
    pad_val: int
        padding index for values array
        Useful when number of values in a row are less than k
    Returns:
    --------
    ind: np.ndarray
        topk indices; size=(num_rows, k)
    val: np.ndarray, optional
        topk val; size=(num_rows, k)
    """
    nr = len(indptr) - 1
    ind = np.full((nr, k), fill_value=pad_ind, dtype=indices.dtype)
    val = np.full((nr, k), fill_value=pad_val, dtype=data.dtype)

    for i in nb.prange(nr):
        s, e = indptr[i], indptr[i+1]
        num_el = min(k, e - s)
        temp = np.argsort(data[s: e])[::-1][:num_el]
        ind[i, :num_el] = indices[s: e][temp]
        val[i, :num_el] = data[s: e][temp]
    return ind, val


def topk(X, k, pad_ind, pad_val, return_values=False,
         dtype='float32', use_cython=False):
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
    dtype: str, optional, default='float32'
        datatype of values
    use_cython: bool, optional, default=False
        use cython to compute topk
        may be helpful when numba isn't working on some machine

    Returns:
    --------
    ind: np.ndarray
        topk indices; size=(num_rows, k)
    val: np.ndarray, optional
        topk val; size=(num_rows, k)
    """
    if use_cython:
        ind, val = _topk(X.data, X.indices, X.indptr, k, pad_ind, pad_val)
    else:
        ind, val = _topk_nb(X.data, X.indices, X.indptr, k, pad_ind, pad_val)
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
    X.sort_indices()
    ranks = rank(X)
    if copy:
        X = X.copy()
    X.data[ranks.data > k] = 0.0
    X.eliminate_zeros()
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


def csr_from_arrays(indices, values, shape=None, dtype='float32'):
    """
    Convert given indices and their corresponding values
    to a csr_matrix

    indices[i, j] => value[i, j]

    Arguments:
    indices: np.ndarray
        array with indices; type should be int
    values: np.ndarray
        array with values
    shape: tuple, optional, default=None
        Infer shape from indices when None
        * Throws error in case of invalid shape
        * Rows in indices and vals must match the given shape
        * Cols in indices must be less than equal to given shape
    """
    assert indices.shape == values.shape, "Shapes for ind and vals must match"
    num_rows, num_cols = indices.shape[0], np.max(indices)+1
    if shape is not None:
        assert num_rows == shape[0], "num_rows_inferred != num_rows_given"
        assert num_cols <= shape[1], "num_cols_inferred > num_cols_given"
    else:
        shape = (num_rows, num_cols)
    # Need the last values (hence +1)
    indptr = np.arange(0, indices.size + 1, indices.shape[1])
    data = values.flatten()
    indices = indices.flatten()
    return csr_matrix((data, indices, indptr), shape=shape)


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
    return sk_normalize(X, norm=norm, copy=copy)


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


def sigmoid(X, copy=False):
    """Sparse sigmoid i.e. zeros are kept intact
    Parameters
    ----------
    X: csr_matrix
        sparse matrix in csr format
    copy: boolean, optional, default=False
        make a copy or not
    """
    if copy:
        X = X.copy()
    X.data = expit(X.data)
    return X


def _map_rows(X, mapping, shape, oformat='csr'):
    """Indices should not be repeated
    Will not convert to dense
    """
    X = X.tocoo()
    row_idx, col_idx, vals = X.row, X.col, X.data 
    func = np.vectorize(lambda x: mapping[x])
    row_idx = func(row_idx)
    return csr_matrix(
        (vals, (row_idx, col_idx)), shape=shape).asformat(oformat)


def _map_cols(X, mapping, shape, oformat='csr'):
    """Indices should not be repeated
    Will not convert to dense
    """
    X = X.tocoo()
    row_idx, col_idx, vals = X.row, X.col, X.data 
    func = np.vectorize(lambda x: mapping[x])
    col_idx = func(col_idx)
    return csr_matrix(
        (vals, (row_idx, col_idx)), shape=shape).asformat(oformat)


def _map(X, mapping, shape, axis=1, oformat='csr'):
    """Map sparse matrix as per given mapping

    Arguments:
    ---------
    X: scipy.sparse matrix
        input matrix
    shape: tuple
        shape of the output matrix
    axis: int, optional, default=1
        1: map columns
        0: map rows 
    oformat: str, optional, default='csr
        'csr' or 'csc' or 'lil' or 'coo'

    Returns:
    -------
        a mapped sparse matrix in the given format
    """
    if oformat not in {'csr', 'csc', 'lil', 'coo'}:
        raise NotImplementedError("Unknown sparse format!")
    if axis == 1:
        return _map_cols(X, mapping, shape, oformat)
    elif axis == 0:
        return _map_rows(X, mapping, shape, oformat)
    else:
        raise NotImplementedError("Unknown axis for sparse matrix!")


def compute_centroid(X, Y, reduction='sum', _binarize=False, copy=True):
    """
    Compute label centroids from sparse features
    * output is sparse

    Arguments:
    ---------
    X: scipy.sparse.csr_matrix
        sparse feature of each document
    Y: scipy.sparse.csr_matrix
        ground truth
    reduction: str, optional (default='sum')
        take sum or average

    Returns:
    --------
    centroids: scipy.sparse.csr_matrix
        Centroid for each label
    """
    if _binarize:
        Y = binarize(Y, copy=copy)
    centroids = Y.T.dot(X).tocsr()
    if reduction == 'sum':
        pass
    elif reduction == 'mean':
        freq = Y.getnnz(axis=0).reshape(-1, 1)
        freq[freq == 0] = 1  # avoid division by zero
        centroids = centroids.multiply(1/freq)
    else:
        raise NotImplementedError(
            "Reduction {} not yet implemented.".format(reduction))
    return centroids
