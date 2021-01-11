import numpy as np
from scipy.sparse import csr_matrix
import warnings
from .dense import topk
from .sparse import csr_from_arrays


class SMatrix(object):
    """
    Sparse matrix (maintain indices and values as dense matrices)

    * Useful when nnz for each row is known in advance
    * Uses n_cols as pad_ind; will remove the pad_ind with .data()

    Arguments:
    ----------
    n_rows: int
        lenght of 0th dimension
    nnz: int
        store nnz values per instance
    n_cols: int
        lenght of 1st dimension
        pad indices with this value as well
    k: int
        the k in top-k
    pad_val: float, optional, default=-1e5
        default value of predictions
    fname: float or None, optional, default=None
        Use memmap files and store on disk if filename is provides
    """
    def __init__(self, n_rows, n_cols, nnz, pad_val=-1e5, fname=None):
        self.n_rows = n_rows
        self.nnz = nnz
        self.n_cols = n_cols
        self.pad_ind = n_cols
        self.pad_val = pad_val
        self.indices = self._array(
            fname + ".ind" if fname is not None else None,
            fill_value=self.pad_ind,
            dtype='int64')
        self.values = self._array(
            fname + ".val" if fname is not None else None,
            fill_value=self.pad_val,
            dtype='float32')

    def _array(self, fname, fill_value, dtype):
        if fname is None:
            arr = np.full(
                (self.n_rows, self.nnz),
                fill_value=fill_value, dtype=dtype)
        else:
            arr = np.memmap(
                fname, shape=(self.n_rows, self.nnz),
                dtype=dtype, mode='w+')
            arr[:] = fill_value
        return arr

    def data(self, format='sparse'):
        """Returns the predictions as a csr_matrix or indices & values arrays
        """
        self.flush()
        if format == 'sparse':
            if not self.in_memory:
                warnings.warn("Files on disk; will create copy in memory.")
            return csr_from_arrays(
                self.indices, self.values,
                shape=(self.n_rows, self.n_cols+1))[:, :-1]
        else:
            return self.indices, self.values

    def update(self, ind, val):
        ind = np.array(ind, dtype='int64')
        vals = np.array(val, dtype='float32')
        self.indices[:] = ind[:]
        self.values[:] = vals[:]
        self.flush()

    def update_block(self, start_idx, ind, val):
        """Update the entries as per given indices and values
        """
        top_ind, top_val = self.topk(ind, val)
        _size = val.shape[0]
        self.values[start_idx: start_idx+_size, :] = top_val
        self.indices[start_idx: start_idx+_size, :] = top_ind

    def topk(self, ind, val):
        """Assumes inputs are np.ndarrays/ Implement your own method
        for some other type.
        Output must be np.ndarrays

        * if ind is None: will return corresponding indices of vals
            typically used with OVA predictions
        * otherwise: will use corresponding entries from ind
            typically used with predictions with a label shortlist
        """
        return topk(val, ind, k=self.nnz)

    @property
    def in_memory(self):
        return not isinstance(self.indices, np.memmap)

    def flush(self):
        if not self.in_memory:
            self.indices.flush()
            self.values.flush()

    #TODO: Unknown behaviour for slicing
    def __getitem__(self, index):
        return self.indices[index], self.values[index]

    def __len__(self):
        return self.n_rows

    def __del__(self):
        del self.indices, self.values

    @property
    def shape(self):
        return self.n_rows, self.n_cols
