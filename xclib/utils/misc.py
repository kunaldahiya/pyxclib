import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.sparse import save_npz
import os


def train_filter_labels(trn_uids, lbl_uids):
    """
        Return train filter labels
        Args:
            trn_uids: Product IDs of train documents
            lbl_uids: Product IDs of labels
    """
    # A = A pairs
    _, x, y = np.intersect1d(trn_uids, lbl_uids, return_indices=True)
    filter_trn = np.vstack([x, y]).T
    return filter_trn


def test_filter_labels(trn_uids, tst_uids, lbl_uids, trn_X_Y):
    """
        Return test filter labels
        Args:
            trn_uids: Product IDs of train 
            tst_uids: Product IDs of test
            lbl_uids: Product IDs of labels
            trn_X_Y: csr matrix of training pairs
    """
    # A = A pairs
    _, x, y = np.intersect1d(tst_uids, lbl_uids, return_indices=True)
    filter_tst_initial = np.vstack([x, y]).T

    # A -> B, B -> A pairs in train-test
    _, x, y = np.intersect1d(trn_uids, lbl_uids, return_indices=True)
    trn_lbl_intersect = np.vstack([x, y]).T

    u_trn_doc, filter_x_trn = np.unique(trn_lbl_intersect[:, 0], return_index=True)
    u_tst_lbl, filter_y_trn = np.unique(filter_tst_initial[:, 1], return_index=True) 
    tst_trn = trn_X_Y[u_trn_doc][:, u_tst_lbl].T
    rows, cols = tst_trn.nonzero()
    rows = filter_tst_initial[:, 0][filter_y_trn[rows]]
    cols = trn_lbl_intersect[:, 1][filter_x_trn[cols]]
    overlap = np.vstack([rows, cols]).T

    filter_tst = np.vstack([filter_tst_initial, overlap])
    return filter_tst


def merge_predictions(pred_0, pred_1, beta):
    return beta*pred_0 + (1-beta)*pred_1


def convert_to_sparse(weight, bias):
    weight = np.vstack(weight).squeeze()
    bias = np.vstack(bias).squeeze()
    return csr_matrix(weight), csr_matrix(bias).transpose()


def _update_predicted(start_idx, predicted_batch_labels, 
                      predicted_labels, top_k=10):
    """
        Update the predicted answers for the batch
        Args:
            predicted_batch_labels
            predicted_labels
    """
    def _select_topk(vec, k):
        batch_size = vec.shape[0]
        top_ind = np.argpartition(vec, -k)[:, -k:]
        ind = np.zeros((k*batch_size, 2), dtype=np.int)
        ind[:, 0] = np.repeat(np.arange(0, batch_size, 1), [k]*batch_size)
        ind[:, 1] = top_ind.flatten('C')
        return top_ind.flatten('C'), vec[ind[:, 0], ind[:, 1]]
    batch_size = predicted_batch_labels.shape[0]
    top_indices, top_vals = _select_topk(predicted_batch_labels, k=top_k)
    ind = np.zeros((top_k*batch_size, 2), dtype=np.int)
    ind[:, 0] = np.repeat(
        np.arange(start_idx, start_idx+batch_size, 1), [top_k]*batch_size)
    ind[:, 1] = top_indices
    predicted_labels[ind[:, 0], ind[:, 1]] = top_vals


def _update_predicted_shortlist(start_idx, predicted_batch_labels,
                                predicted_labels, shortlist, top_k=10):
    """
        Update the predicted answers for the batch
        Args:
            predicted_batch_labels
            predicted_labels
    """
    def _select_topk(vec, k):
        batch_size = vec.shape[0]
        top_ind = np.argpartition(vec, -k)[:, -k:]
        row_idx = np.arange(batch_size).reshape(-1, 1)
        return top_ind, vec[row_idx, top_ind]

    def _select_2d(src, indices):
        n_rows, n_cols = indices.shape
        ind = np.zeros((n_rows*n_cols, 2), dtype=np.int)
        ind[:, 0] = np.repeat(np.arange(n_rows), [n_cols]*n_rows)
        ind[:, 1] = indices.flatten('C')
        return src[ind[:, 0], ind[:, 1]].flatten('C')

    batch_size = predicted_batch_labels.shape[0]
    top_indices, top_values = _select_topk(predicted_batch_labels, top_k)
    ind = np.zeros((top_k*batch_size, 2), dtype=np.int)
    ind[:, 0] = np.repeat(
        np.arange(start_idx, start_idx+batch_size, 1), [top_k]*batch_size)
    ind[:, 1] = _select_2d(shortlist, top_indices)
    vals = top_values.flatten('C')
    predicted_labels[ind[:, 0], ind[:, 1]] = vals


def save_predictions(preds, result_dir, valid_labels, num_samples, 
                     num_labels, _fnames=['knn', 'clf']):
    if isinstance(preds, tuple):
        for _, (_pred, _fname) in enumerate(zip(preds, _fnames)):
            predicted_labels = map_to_original(
                _pred, valid_labels, _shape=(num_samples, num_labels))
            save_npz(os.path.join(
                result_dir, 'predictions_{}.npz'.format(
                    _fname)), predicted_labels)
    else:
        predicted_labels = map_to_original(
            preds, valid_labels, _shape=(num_samples, num_labels))
        save_npz(os.path.join(result_dir, 'predictions.npz'), predicted_labels)
