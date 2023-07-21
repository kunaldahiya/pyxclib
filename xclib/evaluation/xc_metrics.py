"""
    Compute evaluation statistics.
"""
import scipy.sparse as sp
import numpy as np
import warnings
from xclib.utils.sparse import topk, binarize
from xclib.utils.numba_utils import in1d
import numba as nb


def compatible_shapes(x, y):
    """
    See if both matrices have same shape

    Works fine for the following combinations:
    * both are sparse
    * both are dense
    
    Will only compare rows when:
    * one is sparse/dense and other is dict
    * one is sparse and other is dense 

    ** User must ensure that predictions are of correct shape when a
    np.ndarray is passed with all predictions. 
    """
    # both are either sparse or dense
    if (sp.issparse(x) and sp.issparse(y)) \
        or (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
        return x.shape == y.shape

    # compare #rows if one is sparse and other is dict or np.ndarray
    if not (isinstance(x, dict) or isinstance(y, dict)):
        return x.shape[0] == y.shape[0]
    else:
        if isinstance(x, dict):
            return len(x['indices']) == len(x['scores']) == y.shape[0]
        else:
            return len(y['indices']) == len(y['scores']) == x.shape[0]


def jaccard_similarity(pred_0, pred_1, copy=True, y=None):
    """Jaccard similary b/w two different predictions matrices
    Args:
    pred_0: csr_matrix
        prediction for algorithm 0
    pred_1: csr_matrix
        prediction for algorithm 1
    copy: bool
        retain original pred_0 and pred_1 if true
    y: csr_matrix or None
        true labels
    """
    if copy:
        pred_0 = pred_0.copy()
        pred_1 = pred_1.copy()

    def _correct_only(pred, y):
        pred = pred.multiply(y)
        pred.eliminate_zeros()
        return pred

    def _safe_divide(a, b):
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.true_divide(a, b)
            out[out == np.inf] = 0
            return np.nan_to_num(out)

    if y is not None:
        pred_0 = _correct_only(pred_0, y)
        pred_1 = _correct_only(pred_1, y)

    pred_0, pred_1 = binarize(pred_0), binarize(pred_1)
    intersection = np.array(pred_0.multiply(pred_1).sum(axis=1)).ravel()
    union = np.array(binarize(pred_0 + pred_1).sum(axis=1)).ravel()
    return np.mean(_safe_divide(intersection, union))


def format(*args, decimal_points='%0.2f'):
    out = []
    for vals in args:
        out.append(
            ','.join(list(map(lambda x: decimal_points % (x*100), vals))))
    return '\n'.join(out)


def _broad_cast(mat, like):
    if isinstance(like, np.ndarray):
        return np.asarray(mat)
    elif sp.issparse(mat):
        return mat
    else:
        raise NotImplementedError(
            "Unknown type; please pass csr_matrix, np.ndarray or dict.")


def _get_topk_sparse(X, pad_indx=0, k=5, use_cython=False):
    """
    Get top-k elements when X is a sparse matrix
    * Support for cython (use_cython=True) and numba (use_cython=False)
    """
    X = X.tocsr()
    X.sort_indices()
    pad_indx = X.shape[1]
    indices = topk(
        X, k, pad_indx, 0, return_values=False, use_cython=use_cython)
    return indices


def _get_topk_array(X, k=5, sorted=False):
    """
    Get top-k elements when X is an array
    X can be an array of:
        indices: indices of top predictions (must be sorted)
        values: scores for all labels (like in one-vs-all)
    """
    # indices are given
    assert X.shape[1] >= k, "Number of elements in X is < {}".format(k)
    if np.issubdtype(X.dtype, np.integer):
        assert sorted, "sorted must be true with indices"
        indices = X[:, :k] if X.shape[1] > k else X
    # values are given
    elif np.issubdtype(X.dtype, np.floating):
        _indices = np.argpartition(X, -k)[:, -k:]
        _scores = np.take_along_axis(
            X, _indices, axis=-1
        )
        indices = np.argsort(-_scores, axis=-1)
        indices = np.take_along_axis(_indices, indices, axis=1)
    return indices


def _get_topk_dict(X, k=5, sorted=False):
    """
    Get top-k elements when X is an dict of indices and scores
    X['scores'][i, j] will contain score of 
        ith instance and X['indices'][i, j]th label 
    """
    indices = X['indices']
    scores = X['scores']
    assert compatible_shapes(indices, scores), \
        "Dimension mis-match: expected array of shape {} found {}".format(
            indices.shape, scores.shape)
    assert scores.shape[1] >= k, "Number of elements in X is < {}".format(
        k)
    # assumes indices are already sorted by the user
    if sorted:
        return indices[:, :k] if indices.shape[1] > k else indices

    # get top-k entried without sorting them
    if scores.shape[1] > k:
        _indices = np.argpartition(scores, -k)[:, -k:]
        _scores = np.take_along_axis(
            scores, _indices, axis=-1
        )
        # sort top-k entries
        __indices = np.argsort(-_scores, axis=-1)
        _indices = np.take_along_axis(_indices, __indices, axis=-1)
        indices = np.take_along_axis(indices, _indices, axis=-1)
    else:
        _indices = np.argsort(-scores, axis=-1)
        indices = np.take_along_axis(indices, _indices, axis=-1)
    return indices


def _get_topk(X, pad_indx=0, k=5, sorted=False, use_cython=False):
    """
    Get top-k indices (row-wise); Support for
    * csr_matirx
    * 2 np.ndarray with indices and values
    * np.ndarray with indices or values
    """
    if sp.issparse(X):
        indices = _get_topk_sparse(
            X=X,
            pad_indx=pad_indx,
            k=k,
            use_cython=use_cython)
    elif isinstance(X, np.ndarray):
        indices = _get_topk_array(
            X=X,
            k=k,
            sorted=sorted)
    elif isinstance(X, dict):
        indices = _get_topk_dict(
            X=X,
            k=k,
            sorted=sorted)
    else:
        raise NotImplementedError(
            "Unknown type; please pass csr_matrix, np.ndarray or dict.")
    return indices


def compute_inv_propesity(labels, A, B):
    """
    Computes inverse propernsity as proposed in Jain et al. 16.

    Arguments:
    ---------
    labels: csr_matrix
        label matrix (typically ground truth for train data)
    A: float
        typical values:
        * 0.5: Wikipedia
        * 0.6: Amazon
        * 0.55: otherwise
    B: float
        typical values:
        * 0.4: Wikipedia
        * 2.6: Amazon
        * 1.5: otherwise

    Returns:
    -------
    np.ndarray: propensity scores for each label
    """
    num_instances, _ = labels.shape
    freqs = np.ravel(np.sum(labels, axis=0))
    C = (np.log(num_instances)-1)*np.power(B+1, A)
    wts = 1.0 + C*np.power(freqs+B, -A)
    return np.ravel(wts)


def _setup_metric(X, true_labels, inv_psp=None,
                  k=5, sorted=False, use_cython=False):
    assert compatible_shapes(X, true_labels), \
        "ground truth and prediction matrices must have same shape."
    num_instances, num_labels = true_labels.shape
    indices = _get_topk(X, num_labels, k, sorted, use_cython)
    ps_indices = None
    if inv_psp is not None:
        _mat = sp.spdiags(inv_psp, diags=0,
                          m=num_labels, n=num_labels)
        _psp_wtd = _broad_cast(_mat.dot(true_labels.T).T, true_labels)
        ps_indices = _get_topk(_psp_wtd, num_labels, k, False, use_cython)
        inv_psp = np.hstack([inv_psp, np.zeros((1))])

    idx_dtype = true_labels.indices.dtype
    true_labels = sp.csr_matrix(
        (true_labels.data, true_labels.indices, true_labels.indptr),
        shape=(num_instances, num_labels+1), dtype=true_labels.dtype)

    # scipy won't respect the dtype of indices
    # may fail otherwise on really large datasets
    true_labels.indices = true_labels.indices.astype(idx_dtype)
    return indices, true_labels, ps_indices, inv_psp


def _eval_flags(indices, true_labels, inv_psp=None):
    if sp.issparse(true_labels):
        nr, nc = indices.shape
        rows = np.repeat(np.arange(nr).reshape(-1, 1), nc)
        eval_flags = true_labels[rows, indices.ravel()].A1.reshape(nr, nc)
    elif type(true_labels) == np.ndarray:
        eval_flags = np.take_along_axis(true_labels,
                                        indices, axis=-1)
    if inv_psp is not None:
        eval_flags = np.multiply(inv_psp[indices], eval_flags)
    return eval_flags


def precision(X, true_labels, k=5, sorted=False, use_cython=False):
    """
    Compute precision@k for 1-k

    Arguments:
    ----------
    X: csr_matrix, np.ndarray or dict
        * csr_matrix: csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sorted order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}

    true_labels: csr_matrix or np.ndarray
        ground truth in sparse or dense format
    k: int, optional (default=5)
        compute precision till k
    sorted: boolean, optional, default=False
        whether X is already sorted (will skip sorting)
        * used when X is of type dict or np.ndarray (of indices)
        * shape is not checked is X are np.ndarray
        * must be set to true when X are np.ndarray (of indices)
    use_cython: boolean, optional, default=False
        whether to use cython version to find top-k element
        * defaults to numba version
        * may be useful when numba version fails on a system

    Returns:
    -------
    np.ndarray: precision values for 1-k
    """
    indices, true_labels, _, _ = _setup_metric(
        X, true_labels, k=k, sorted=sorted, use_cython=use_cython)
    eval_flags = _eval_flags(indices, true_labels, None)
    return _precision(eval_flags, k)


def psprecision(X, true_labels, inv_psp, k=5, sorted=False, use_cython=False):
    """
    Compute propensity scored precision@k for 1-k

    Arguments:
    ----------
    X: csr_matrix, np.ndarray or dict
        * csr_matrix: csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sorted order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}
    true_labels: csr_matrix or np.ndarray
        ground truth in sparse or dense format
    inv_psp: np.ndarray
        propensity scores for each label
    k: int, optional (default=5)
        compute propensity scored precision till k
    sorted: boolean, optional, default=False
        whether X is already sorted (will skip sorting)
        * used when X is of type dict or np.ndarray (of indices)
        * shape is not checked is X are np.ndarray
        * must be set to true when X are np.ndarray (of indices)
    use_cython: boolean, optional, default=False
        whether to use cython version to find top-k element
        * defaults to numba version
        * may be useful when numba version fails on a system

    Returns:
    -------
    np.ndarray: propensity scored precision values for 1-k
    """
    indices, true_labels, ps_indices, inv_psp = _setup_metric(
        X, true_labels, inv_psp, k=k, sorted=sorted, use_cython=use_cython)
    eval_flags = _eval_flags(indices, true_labels, inv_psp)
    ps_eval_flags = _eval_flags(ps_indices, true_labels, inv_psp)
    return _precision(eval_flags, k)/_precision(ps_eval_flags, k)


def _precision(eval_flags, k=5):
    deno = 1/(np.arange(k)+1)
    precision = np.mean(
        np.multiply(np.cumsum(eval_flags, axis=-1), deno),
        axis=0)
    return np.ravel(precision)


def ndcg(X, true_labels, k=5, sorted=False, use_cython=False):
    """
    Compute nDCG@k for 1-k

    Arguments:
    ----------
    X: csr_matrix, np.ndarray or dict
        * csr_matrix: csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sorted order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}
    true_labels: csr_matrix or np.ndarray
        ground truth in sparse or dense format
    k: int, optional (default=5)
        compute nDCG till k
    sorted: boolean, optional, default=False
        whether X is already sorted (will skip sorting)
        * used when X is of type dict or np.ndarray (of indices)
        * shape is not checked is X are np.ndarray
        * must be set to true when X are np.ndarray (of indices)
    use_cython: boolean, optional, default=False
        whether to use cython version to find top-k element
        * defaults to numba version
        * may be useful when numba version fails on a system


    Returns:
    -------
    np.ndarray: nDCG values for 1-k
    """
    indices, true_labels, _, _ = _setup_metric(
        X, true_labels, k=k, sorted=sorted, use_cython=use_cython)
    eval_flags = _eval_flags(indices, true_labels, None)
    _total_pos = np.asarray(
        true_labels.sum(axis=1),
        dtype=np.int32)
    _max_pos = max(np.max(_total_pos), k)
    _cumsum = np.cumsum(1/np.log2(np.arange(1, _max_pos+1)+1))
    n = _cumsum[_total_pos - 1]
    return _ndcg(eval_flags, n, k)


def psndcg(X, true_labels, inv_psp, k=5, sorted=False, use_cython=False):
    """
    Compute propensity scored nDCG@k for 1-k

    Arguments:
    ----------
    X: csr_matrix, np.ndarray or dict
        * csr_matrix: csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sorted order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}
    true_labels: csr_matrix or np.ndarray
        ground truth in sparse or dense format
    inv_psp: np.ndarray
        propensity scores for each label
    k: int, optional (default=5)
        compute propensity scored nDCG till k
    sorted: boolean, optional, default=False
        whether X is already sorted (will skip sorting)
        * used when X is of type dict or np.ndarray (of indices)
        * shape is not checked is X are np.ndarray
        * must be set to true when X are np.ndarray (of indices)
    use_cython: boolean, optional, default=False
        whether to use cython version to find top-k element
        * defaults to numba version
        * may be useful when numba version fails on a system


    Returns:
    -------
    np.ndarray: propensity scored nDCG values for 1-k
    """
    indices, true_labels, ps_indices, inv_psp = _setup_metric(
        X, true_labels, inv_psp, k=k, sorted=sorted, use_cython=use_cython)
    eval_flags = _eval_flags(indices, true_labels, inv_psp)
    ps_eval_flags = _eval_flags(ps_indices, true_labels, inv_psp)
    _total_pos = np.asarray(
        true_labels.sum(axis=1),
        dtype=np.int32)
    _max_pos = max(np.max(_total_pos), k)
    _cumsum = np.cumsum(1/np.log2(np.arange(1, _max_pos+1)+1))
    n = _cumsum[_total_pos - 1]
    return _ndcg(eval_flags, n, k)/_ndcg(ps_eval_flags, n, k)


def _ndcg(eval_flags, n, k=5):
    _cumsum = 0
    _dcg = np.cumsum(np.multiply(
        eval_flags, 1/np.log2(np.arange(k)+2)),
        axis=-1)
    ndcg = np.zeros((1, k), dtype=np.float32)
    for _k in range(k):
        _cumsum += 1/np.log2(_k+1+1)
        ndcg[0, _k] = np.mean(
            np.multiply(_dcg[:, _k].reshape(-1, 1), 1/np.minimum(n, _cumsum))
        )
    return np.ravel(ndcg)


def recall(X, true_labels, k=5, sorted=False, use_cython=False):
    """
    Compute recall@k for 1-k

    Arguments:
    ----------
    X: csr_matrix, np.ndarray or dict
        * csr_matrix: csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sorted order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}
    true_labels: csr_matrix or np.ndarray
        ground truth in sparse or dense format
    k: int, optional (default=5)
        compute recall till k
    sorted: boolean, optional, default=False
        whether X is already sorted (will skip sorting)
        * used when X is of type dict or np.ndarray (of indices)
        * shape is not checked is X are np.ndarray
        * must be set to true when X are np.ndarray (of indices)
    use_cython: boolean, optional, default=False
        whether to use cython version to find top-k element
        * defaults to numba version
        * may be useful when numba version fails on a system


    Returns:
    -------
    np.ndarray: recall values for 1-k
    """
    indices, true_labels, _, _ = _setup_metric(
        X, true_labels, k=k, sorted=sorted, use_cython=use_cython)
    deno = true_labels.sum(axis=1)
    deno[deno == 0] = 1
    deno = 1/deno
    eval_flags = _eval_flags(indices, true_labels, None)
    return _recall(eval_flags, deno, k)


def hits(X, true_labels, k=5, sorted=False, use_cython=False):
    """
    Compute hits@k for 1-k

    Arguments:
    ----------
    X: csr_matrix, np.ndarray or dict
        * csr_matrix: csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sorted order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}
    true_labels: csr_matrix or np.ndarray
        ground truth in sparse or dense format
    k: int, optional (default=5)
        compute recall till k
    sorted: boolean, optional, default=False
        whether X is already sorted (will skip sorting)
        * used when X is of type dict or np.ndarray (of indices)
        * shape is not checked is X are np.ndarray
        * must be set to true when X are np.ndarray (of indices)
    use_cython: boolean, optional, default=False
        whether to use cython version to find top-k element
        * defaults to numba version
        * may be useful when numba version fails on a system


    Returns:
    -------
    np.ndarray: hits values for 1-k
    """
    indices, true_labels, _, _ = _setup_metric(
        X, true_labels, k=k, sorted=sorted, use_cython=use_cython)
    eval_flags = _eval_flags(indices, true_labels, None)
    return _hits(eval_flags)


def _hits(eval_flags):
    eval_flags = np.clip(np.cumsum(eval_flags, axis=-1), 0, 1)
    hits = np.mean(eval_flags, axis=0)
    return np.ravel(hits)


def psrecall(X, true_labels, inv_psp, k=5, sorted=False, use_cython=False):
    """
    Compute propensity scored recall@k for 1-k

    Arguments:
    ----------
    X: csr_matrix, np.ndarray or dict
        * csr_matrix: csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sorted order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}
    true_labels: csr_matrix or np.ndarray
        ground truth in sparse or dense format
    inv_psp: np.ndarray
        propensity scores for each label
    k: int, optional (default=5)
        compute propensity scored recall till k
    sorted: boolean, optional, default=False
        whether X is already sorted (will skip sorting)
        * used when X is of type dict or np.ndarray (of indices)
        * shape is not checked is X are np.ndarray
        * must be set to true when X are np.ndarray (of indices)
    use_cython: boolean, optional, default=False
        whether to use cython version to find top-k element
        * defaults to numba version
        * may be useful when numba version fails on a system


    Returns:
    -------
    np.ndarray: propensity scored recall values for 1-k
    """
    indices, true_labels, ps_indices, inv_psp = _setup_metric(
        X, true_labels, inv_psp, k=k, sorted=sorted, use_cython=use_cython)
    deno = true_labels.sum(axis=1)
    deno[deno == 0] = 1
    deno = 1/deno
    eval_flags = _eval_flags(indices, true_labels, inv_psp)
    ps_eval_flags = _eval_flags(ps_indices, true_labels, inv_psp)
    return _recall(eval_flags, deno, k)/_recall(ps_eval_flags, deno, k)


def _recall(eval_flags, deno, k=5):
    eval_flags = np.cumsum(eval_flags, axis=-1)
    recall = np.mean(np.multiply(eval_flags, deno), axis=0)
    return np.ravel(recall)


def _auc(X, k):
    non_inv = np.cumsum(X, axis=1)
    cum_noninv = np.sum(np.multiply(non_inv, 1-X), axis=1)
    n_pos = non_inv[:, -1]
    all_pairs = np.multiply(n_pos, k-n_pos)
    all_pairs[all_pairs == 0] = 1.0  # for safe divide
    point_auc = np.divide(cum_noninv, all_pairs)
    return np.mean(point_auc)


def auc(X, true_labels, k, sorted=False, use_cython=False):
    """
    Compute AUC score

    Arguments:
    ----------
    X: csr_matrix, np.ndarray or dict
        * csr_matrix: csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sorted order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}
    true_labels: csr_matrix or np.ndarray
        ground truth in sparse or dense format
    k: int, optional (default=5)
        retain top-k predictions only
    sorted: boolean, optional, default=False
        whether X is already sorted (will skip sorting)
        * used when X is of type dict or np.ndarray (of indices)
        * shape is not checked is X are np.ndarray
        * must be set to true when X are np.ndarray (of indices)
    use_cython: boolean, optional, default=False
        whether to use cython version to find top-k element
        * defaults to numba version
        * may be useful when numba version fails on a system


    Returns:
    -------
    np.ndarray: auc score
    """
    indices, true_labels, _, _ = _setup_metric(
        X, true_labels, k=k, sorted=sorted, use_cython=use_cython)
    eval_flags = _eval_flags(indices, true_labels, None)
    return _auc(eval_flags, k)


class Metrics(object):
    def __init__(self, true_labels, inv_psp=None, remove_invalid=False):
        """
        Class to compute vanilla and propensity scored precision and ndcg

        Arguments:
        ---------
        true_labels: csr_matrix or np.ndarray
            ground truth in sparse or dense format
            shape: (num_instances, num_labels)
        inv_psp: np.ndarray or None; default=None
            propensity scores for each label
            will compute propensity scores only with valid values
        remove_invalid: boolean; default=False
            Remove test samples without any positive label
        """
        self.true_labels = true_labels
        self.num_instances, self.num_labels = true_labels.shape
        self.remove_invalid = remove_invalid
        self.valid_idx = None
        if self.remove_invalid:
            samples = np.sum(self.true_labels, axis=1)
            self.valid_idx = np.arange(
                self.num_instances).reshape(-1, 1)[samples > 0]
            self.true_labels = self.true_labels[self.valid_idx]
            self.num_instances = self.valid_idx.size
        "inserting dummpy index"
        self.ndcg_denominator = np.cumsum(
            1/np.log2(np.arange(1, self.num_labels+1)+1))
        self.inv_psp = None
        if inv_psp is not None:
            self.inv_psp = np.ravel(inv_psp)

    def eval(self, pred_labels, K=5, sorted=False, use_cython=False):
        """
        Compute values

        Arguments:
        ---------
        predicted_labels: csr_matrix, np.ndarray, dict
            * csr_matrix: csr_matrix with nnz at relevant places
            * np.ndarray (float): scores for each label
              User must ensure shape is fine
            * np.ndarray (int): top indices (in sorted order)
              User must ensure shape is fine
            * {'indices': np.ndarray, 'scores': np.ndarray}
        k: int, optional; default=5
            compute values till k
        sorted: boolean, optional, default=False
            whether pred_labels is already sorted (will skip sorting)
            * used when pred_labels is of type dict or np.ndarray (of indices)
            * shape is not checked is pred_labels are np.ndarray
            * must be set to true when pred_labels are np.ndarray (of indices)
        use_cython: boolean, optional, default=False
            whether to use cython version to find top-k element
            * defaults to numba version
            * may be useful when numba version fails on a system
        Returns:
        -------
        list: vanilla metrics if inv_psp is None
            vanilla and propensity scored metrics, otherwise
        """
        if self.valid_idx is not None:
            if isinstance(pred_labels, dict):
                pred_labels['indices'] = pred_labels['indices'][self.valid_idx]
                pred_labels['scores'] = pred_labels['scores'][self.valid_idx]
            else:
                pred_labels = pred_labels[self.valid_idx]
            
        assert compatible_shapes(self.true_labels, pred_labels), \
            "Shapes must be compatible for ground truth and predictions"
        indices, true_labels, ps_indices, inv_psp = _setup_metric(
            pred_labels, self.true_labels,
            self.inv_psp, k=K, sorted=sorted,
            use_cython=use_cython)
        _total_pos = np.asarray(
            true_labels.sum(axis=1),
            dtype=np.int32)
        n = self.ndcg_denominator[_total_pos - 1]
        eval_flags = _eval_flags(indices, true_labels, None)
        prec = _precision(eval_flags, K)
        ndcg = _ndcg(eval_flags, n, K)
        if self.inv_psp is not None:
            eval_flags = np.multiply(inv_psp[indices], eval_flags)
            ps_eval_flags = _eval_flags(ps_indices, true_labels, inv_psp)
            PSprec = _precision(eval_flags, K)/_precision(ps_eval_flags, K)
            PSnDCG = _ndcg(eval_flags, n, K)/_ndcg(ps_eval_flags, n, K)
            return [prec, ndcg, PSprec, PSnDCG]
        else:
            return [prec, ndcg]


class Metrices(Metrics):
    def __init__(self, true_labels, inv_propensity_scores=None,
                 remove_invalid=False):
        warnings.warn(
            "Metrices() is deprecated; use Metrics().",
            category=FutureWarning)
        super().__init__(true_labels, inv_propensity_scores, remove_invalid)

@nb.njit(parallel=True)
def restict_preds_for_gt_calc(pred_indices, num_gt, pad_val):
    """
    Returns top GT indices for each data point
    Arguments:
    ----------
    X_indices: np.ndarray
        2D numpy array with indices sorted in descending order according to score for each row
    num_gt: np.array
        number of ground truth for each element
    k: int
        max number of preds to consider
    Returns:
    -------
    np.ndarray:  top min(GT, k) indices for each data point
    """
    num_docs = pred_indices.shape[0]
    restricted_preds = pred_indices + pad_val # ensures that indices that are not overwritten will not intersect with any GT
    max_preds = pred_indices.shape[1]
    for doc_indx in nb.prange(num_docs):
        restrict_indx = min(max_preds, num_gt[doc_indx])
        restricted_preds[doc_indx][: restrict_indx] = pred_indices[doc_indx][:restrict_indx]
    return restricted_preds

@nb.njit(parallel=True)
def _micro_recall_at_gt(pred_indices, true_indices, true_indptr):
    m = pred_indices.shape[0]
    intersections = np.zeros(m, dtype=np.float64)
    gt_cardinality = np.zeros(m, dtype= np.float64)
    for i in nb.prange(pred_indices.shape[0]):
        intersections[i] = np.sum(in1d(pred_indices[i], np.unique(true_indices[true_indptr[i]: true_indptr[i + 1]])))
        gt_cardinality[i] = true_indptr[i + 1] - true_indptr[i]

    micro_recall = np.sum(intersections) / np.sum(gt_cardinality)
    return micro_recall

@nb.njit(parallel=True)
def _recall_at_gt(pred_indices, true_indices, true_indptr):
    m = pred_indices.shape[0]
    intersections = np.zeros(m, dtype=np.float64)
    gt_cardinality = np.zeros(m, dtype= np.float64)
    for i in nb.prange(pred_indices.shape[0]):
        intersections[i] = np.sum(in1d(pred_indices[i], np.unique(true_indices[true_indptr[i]: true_indptr[i + 1]])))
        gt_cardinality[i] = true_indptr[i + 1] - true_indptr[i]
    gt_cardinality[gt_cardinality == 0] = 1
    recall = np.mean(np.multiply(intersections, 1 / gt_cardinality))
    return recall


def process_indices(pred_indices, true_labels, max_preds):
    '''
    Ensure that there are no duplicate indices for each data point _get_topk will pad with the index num_labels for remaining slots, this will be problematic in in1d later
    '''
    num_docs, num_labels = true_labels.shape
    offsets = np.arange(max_preds) + 1
    padded_indices = np.where(pred_indices == num_labels - 1)
    
    offset_arr = np.zeros((num_docs, max_preds))

    offset_arr[padded_indices] = 1
    pred_indices = pred_indices + offset_arr * offsets
    return pred_indices


def micro_recall_at_gt(X, true_labels, pad_val, sorted=False, use_cython=False):
    """
    Compute MicroRecall@GT 

    Arguments:
    ----------
    X: csr_matrix, np.ndarray
        * csr_matrix: csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sorted order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}
    true_labels: csr_matrix or np.ndarray
        ground truth in sparse or dense format
    sorted: boolean, optional, default=False
        whether X is already sorted (will skip sorting)
        * used when X is of type dict or np.ndarray (of indices)
        * shape is not checked is X are np.ndarray
        * must be set to true when X are np.ndarray (of indices)
    use_cython: boolean, optional, default=False
        whether to use cython version to find top-k element
        * defaults to numba version
        * may be useful when numba version fails on a system

    Returns:
    -------
    np.float32: MicroRecall@GT
    """
    if sp.issparse(X):
        max_preds = np.max(X.getnnz(axis = 1))
    else: # numpy array
        max_preds = X.shape[1]
        
    indices, true_labels, _, _ = _setup_metric(
        X, true_labels, k=max_preds, sorted=sorted, use_cython=use_cython)
    num_gt = true_labels.getnnz(axis = 1)
    indices = process_indices(indices, true_labels, max_preds)
    top_gt_indices = restict_preds_for_gt_calc(indices, num_gt, pad_val)
    return _micro_recall_at_gt(top_gt_indices, true_labels.indices.astype(np.int64), true_labels.indptr)

def recall_at_gt(X, true_labels, pad_val, sorted=False, use_cython=False):
    """
    Compute Recall@GT 

    Arguments:
    ----------
    X: csr_matrix, np.ndarray
        * csr_matrix: csr_matrix with nnz at relevant places
        * np.ndarray (float): scores for each label
            User must ensure shape is fine
        * np.ndarray (int): top indices (in sorted order)
            User must ensure shape is fine
        * {'indices': np.ndarray, 'scores': np.ndarray}
    true_labels: csr_matrix or np.ndarray
        ground truth in sparse or dense format
    sorted: boolean, optional, default=False
        whether X is already sorted (will skip sorting)
        * used when X is of type dict or np.ndarray (of indices)
        * shape is not checked is X are np.ndarray
        * must be set to true when X are np.ndarray (of indices)
    use_cython: boolean, optional, default=False
        whether to use cython version to find top-k element
        * defaults to numba version
        * may be useful when numba version fails on a system

    Returns:
    -------
    np.float32: Recall@GT
    """
    if sp.issparse(X):
        max_preds = np.max(X.getnnz(axis = 1))
    else: # numpy array
        max_preds = X.shape[1]
    indices, true_labels, _, _ = _setup_metric(
        X, true_labels, k=max_preds, sorted=sorted, use_cython=use_cython)
    num_gt = true_labels.getnnz(axis = 1)
    indices = process_indices(indices, true_labels, max_preds)
    top_gt_indices = restict_preds_for_gt_calc(indices, num_gt, pad_val)
    return _recall_at_gt(top_gt_indices, true_labels.indices.astype(np.int64), true_labels.indptr)

@nb.njit(parallel=True)
def _recall_at_k(true_labels_indices, true_labels_indptr, pred_labels_data, pred_labels_indices, pred_labels_indptr, top_k):
    fracs = -1 * np.ones((len(true_labels_indptr) - 1, ), dtype=np.float32)
    for i in nb.prange(len(true_labels_indptr) - 1):
        _true_labels = true_labels_indices[true_labels_indptr[i] : true_labels_indptr[i + 1]]
        _data = pred_labels_data[pred_labels_indptr[i] : pred_labels_indptr[i + 1]]
        _indices = pred_labels_indices[pred_labels_indptr[i] : pred_labels_indptr[i + 1]]
        top_inds = np.argsort(_data)[::-1][:top_k]
        _pred_labels = _indices[top_inds]
        if(len(_true_labels) > 0):
            fracs[i] = len(set(_pred_labels).intersection(set(_true_labels))) / len(_true_labels)
    return np.mean(fracs[fracs != -1])

def recall_at_k(X, true_labels, k):
    """
    Compute recall@k, faster than using `recall`
    Arguments:
    ----------
    X: csr_matrix
        * csr_matrix: csr_matrix with nnz at relevant places
    true_labels: csr_matrix
        ground truth in sparse format
    k: int
        compute recall at k
    Returns:
    -------
    float: recall@k
    """
    return _recall_at_k(true_labels.indices.astype(np.int64), true_labels.indptr, 
    X.data, X.indices.astype(np.int64), X.indptr, k)
