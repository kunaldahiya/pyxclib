"""
    Compute evaluation statistics.
"""
import scipy.sparse as sp
import numpy as np
from xclib.utils.sparse import topk, binarize, retain_topk
__author__ = 'AM'


def jaccard_similarity(pred_0, pred_1, y=None):
    """Jaccard similary b/w two different predictions matrices
    Args:
    pred_0: csr_matrix
        prediction for algorithm 0
    pred_1: csr_matrix
        prediction for algorithm 1
    y: csr_matrix or None
        true labels
    """
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
        out.append(','.join(list(map(lambda x: decimal_points % (x*100), vals))))
    return '\n'.join(out)


def _get_topk(X, pad_indx=0, k=5):
    if sp.issparse(X):
        X = X.tocsr()
        X.sort_indices()
        pad_indx = X.shape[1]
        indices = topk(X, k, pad_indx, 0, return_values=False)
    elif type(X) == np.ndarray:
        if np.issubdtype(X.dtype, np.integer):
            warnings.warn("Assuming indices are sorted")
            indices = X[:, :k]
        elif np.issubdtype(X.dtype, np.float):
            _indices = np.argpartition(X, -k)[:, -k:]
            _scores = np.take_along_axis(
                X, _indices, axis=-1
            )
            indices = np.argsort(_scores, axis=-1)
            indices = np.take_along_axis(_indices, indices, axis=1)
    elif type(X) == dict:
        indices = X['indices']
        scores = X['scores']
        assert indices.shape == scores.shape, \
            "Dimension mis-match: expected array of shape {} found {}".format(
                indices.shape, scores.shape
            )
        assert scores.shape[1] < k, "Number of elements in X is < {}".format(
            k)
        if scores.shape[1] >= k:
            _indices = np.argpartition(_scores, -k)[:, -k:]
            scores = np.take_along_axis(
                X, _indices, axis=-1
            )
            __indices = np.argsort(scores, axis=-1)
            _indices = np.take_along_axis(_indices, __indices, axis=-1)
            indices = np.take_along_axis(indices, _indices, axis=-1)
    return indices


def compute_inv_propesity(labels, A, B):
    num_instances, _ = labels.shape
    freqs = np.ravel(np.sum(labels, axis=0))
    C = (np.log(num_instances)-1)*np.power(B+1, A)
    wts = 1.0 + C*np.power(freqs+B, -A)
    return np.ravel(wts)


def _setup_metric(X, true_labels, inv_psp=None, k=5):
    num_instances, num_labels = true_labels.shape
    indices = _get_topk(X, num_labels, k)
    ps_indices = None
    if inv_psp is not None:
        ps_indices = _get_topk(
            true_labels.dot(
                sp.spdiags(inv_psp, diags=0,
                           m=num_labels, n=num_labels)),
            num_labels, k)
        inv_psp = np.hstack([inv_psp, np.zeros((1))])

    true_labels = sp.hstack([true_labels,
                             sp.lil_matrix((num_instances, 1),
                                           dtype=np.int32)]).tocsr()
    return indices, true_labels, ps_indices, inv_psp


def _eval_flags(indices, true_labels, inv_psp=None):
    if sp.issparse(true_labels):
        eval_flags = np.take_along_axis(true_labels.tocsc(),
                                        indices, axis=-1).todense()
    elif type(true_labels) == np.ndarray:
        eval_flags = np.take_along_axis(true_labels,
                                        indices, axis=-1)
    if inv_psp is not None:
        eval_flags = np.multiply(inv_psp[indices], eval_flags)
    return eval_flags


def precision(X, true_labels, k=5):
    indices, true_labels, _, _ = _setup_metric(X, true_labels, k=k)
    eval_flags = _eval_flags(indices, true_labels, None)
    return _precision(eval_flags, k)


def psprecision(X, true_labels, inv_psp, k=5):
    indices, true_labels, ps_indices, inv_psp = _setup_metric(
        X, true_labels, inv_psp, k=k)
    eval_flags = _eval_flags(indices, true_labels, inv_psp)
    ps_eval_flags = _eval_flags(ps_indices, true_labels, inv_psp)
    return _precision(eval_flags, k)/_precision(ps_eval_flags, k)


def _precision(eval_flags, k=5):
    deno = 1/(np.arange(k)+1)
    precision = np.mean(
        np.multiply(np.cumsum(eval_flags, axis=-1), deno),
        axis=0)
    return np.ravel(precision)


def ndcg(X, true_labels, k=5):
    indices, true_labels, _, _ = _setup_metric(X, true_labels, k=k)
    eval_flags = _eval_flags(indices, true_labels, None)
    _total_pos = np.asarray(
        true_labels.sum(axis=1),
        dtype=np.int32)
    _max_pos = max(np.max(_total_pos), k)
    _cumsum = np.cumsum(1/np.log2(np.arange(1, _max_pos+1)+1))
    n = _cumsum[_total_pos - 1]
    return _ndcg(eval_flags, n, k)


def psndcg(X, true_labels, inv_psp, k=5):
    indices, true_labels, ps_indices, inv_psp = _setup_metric(
        X, true_labels, inv_psp, k=k)
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
            np.multiply(_dcg[:, _k], 1/np.minimum(n, _cumsum))
        )
    return np.ravel(ndcg)


def recall(X, true_labels, k=5):
    indices, true_labels, _, _ = _setup_metric(X, true_labels, k=k)
    deno = true_labels.sum(axis=1)
    deno[deno == 0] = 1
    deno = 1/deno
    eval_flags = _eval_flags(indices, true_labels, None)
    return _recall(eval_flags, deno, k)


def psrecall(X, true_labels, inv_psp, k=5):
    indices, true_labels, ps_indices, inv_psp = _setup_metric(
        X, true_labels, inv_psp, k=k)
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


class Metrices(object):
    def __init__(self, true_labels, inv_propensity_scores=None, remove_invalid=False, batch_size=20):
        """
            Args:
                true_labels: csr_matrix: true labels with shape (num_instances, num_labels)
                remove_invalid: boolean: remove samples without any true label
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
            1/np.log2(np.arange(1, self.num_labels+1)+1)).reshape(-1, 1)
        self.labels_documents = np.ravel(
            np.array(np.sum(self.true_labels, axis=1), np.int32))
        self.labels_documents[self.labels_documents == 0] = 1
        self.inv_propensity_scores = np.ravel(inv_propensity_scores)
        self.batch_size = batch_size

    def eval(self, predicted_labels, K=5):
        """
            Args:
                predicted_labels: csr_matrix: predicted labels with shape (num_instances, num_labels)
                K: int: compute values from 1-5
        """
        if self.valid_idx is not None:
            predicted_labels = predicted_labels[self.valid_idx]
        assert predicted_labels.shape == self.true_labels.shape
        indices, true_labels, ps_indices, inv_psp = _setup_metric(
            predicted_labels, self.true_labels,
            self.inv_propensity_scores, k=K)
        _total_pos = np.asarray(
            true_labels.sum(axis=1),
            dtype=np.int32)
        _max_pos = max(np.max(_total_pos), K)
        _cumsum = np.cumsum(1/np.log2(np.arange(1, _max_pos+1)+1))
        n = _cumsum[_total_pos - 1]
        eval_flags = _eval_flags(indices, true_labels, None)
        prec = _precision(eval_flags, K)
        ndcg = _ndcg(eval_flags, n, K)
        if self.inv_propensity_scores is not None:
            eval_flags = np.multiply(inv_psp[indices], eval_flags)
            ps_eval_flags = _eval_flags(ps_indices, true_labels, inv_psp)
            PSprec = _precision(eval_flags, K)/_precision(ps_eval_flags, K)
            PSnDCG = _ndcg(eval_flags, n, K)/_ndcg(ps_eval_flags, n, K)
            return [prec, ndcg, PSprec, PSnDCG]
        else:
            return [prec, ndcg]
