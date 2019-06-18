from ._sparse import rank, read_file
from scipy.sparse import csr_matrix, isspmatrix
import numpy as np
import warnings
from sklearn.preprocessing import normalize


def rankdata(X):
    '''
        Rank of each element in decreasing order (per-row)
        Ranking will start from one (with zero at zero entries)
    '''
    ranks = rank(X.data, X.indices, X.indptr)
    return csr_matrix((ranks, X.indices, X.indptr), shape=X.shape)


def retain_topk(X, copy=True, k=5):
    """
        Retain topk values of each row and make everything else zero
        Args:
            X: csr_matrix: sparse mat
            copy: boolean: copy data or change original array
            k: int: retain these many values
        Returns:
            X: csr_matrix: sparse mat with only k entries in each row
    """
    ranks = rankdata(X)
    if copy:
        X = X.copy()
    X[ranks > k] = 0.0
    X.eliminate_zeros()
    return X


def binarize(X, copy=False):
    """
        Binarize a sparse matrix
    """
    X.data[:] = 1.0
    return X


def gen_shape(indices, indptr, zero_based=True):
    _min = min(indices)
    if not zero_based:
        indices = list(map(lambda x: x-_min, indices))
    num_cols = max(indices)
    num_rows = len(indptr) - 1
    return (num_rows, num_cols)


def expand_indptr(num_rows_inferred, num_rows, indptr):
    """
        Expand indptr if inferred num_rows is less than given
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
    """
        Convert a list of list of tuples to a csr_matrix
        Args:
            X: list of list of tuples: nnz indices for each row
            dtype: 'str': for data
            zero_based: boolean or "auto": indices are zero based or not
            shape: tuple: given shape 
        Returns:
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
    return csr_matrix((np.array(data, dtype=dtype), np.array(indices), np.array(indptr)), shape=shape)


def ll_to_sparse(X, shape=None, dtype='float32', zero_based=True):
    """
        Convert a list of list to a csr_matrix; All values are 1.0
        Args:
            X: list of list: nnz indices for each row
            dtype: 'str': for data
            zero_based: boolean or "auto": indices are zero based or not
            shape: tuple: given shape 
        Returns:
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
    return csr_matrix((np.array(data, dtype=dtype), np.array(indices), np.array(indptr)), shape=shape)


def normalize_data(X, norm='l2', copy=True):
    """
        Normalize sparse or dense matrix
        Args:
            X: sparse or dense/matrix
            norm: normalize with l1/l2
            copy: whether to copy data or not
    """
    features = normalize(X, norm=norm, copy=copy)
    return features


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
        actual_dtype, data, ind, indptr, query = \
            read_file(f, dtype, zero_based, query_id,
                      offset, length)
    else:
        with _gen_open(f) as f:
            f, _header_shape = _handle_header(f, header)
            actual_dtype, data, ind, indptr, query = \
                read_file(f, dtype, zero_based, query_id,
                          offset, length)

    data = np.frombuffer(data, actual_dtype)
    indices = np.frombuffer(ind, np.int64)
    indptr = np.frombuffer(indptr, dtype=np.int64)   # never empty
    query = np.frombuffer(query, np.int64)
    data = np.asarray(data, dtype=dtype)    # no-op for float{32,64}
    return data, indices, indptr, query, _header_shape


def _gen_open(f, _mode='rb'):
    # Update this for more generic file types
    return open(f, _mode)
