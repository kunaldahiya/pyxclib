"""
example to show decile wise plot for different methods 
python3 decile_wise_analysis.py data/LF-AmazonTitles-131K/trn_X_Y.txt data/LF-AmazonTitles-131K/tst_X_Y.txt data/LF-AmazonTitles-131K/filter_labels_test.txt score_mats/NGAME.npz,score_mats/ECLARE.npz,../tmp/score_mats/XR-Transformers.npz 5 5
"""

import os
import sys
from xclib.data import data_utils
import numpy as np
from xclib.utils.analysis import decile_contribution_plot


def get_filter_map(fname):
    if fname is not None:
        return np.loadtxt(fname).astype(np.int)
    else:
        return None


def filter_predictions(pred, mapping):
    if mapping is not None and len(mapping) > 0:
        print("Filtering labels.")
        pred[mapping[:, 0], mapping[:, 1]] = 0
        pred.eliminate_zeros()
    return pred


def main():
    trn_labels = data_utils.read_sparse_file(sys.argv[1])
    tst_labels = data_utils.read_sparse_file(sys.argv[2])
    mapping = get_filter_map(sys.argv[3])

    predictions = dict()

    for n in sys.argv[4].split(","):
        print(f"Reading {n}")
        k = os.path.basename(n).split(".")[0]
        v = filter_predictions(data_utils.read_gen_sparse(n), mapping)
        predictions[k] = v

    num_splits = int(sys.argv[5])
    k = int(sys.argv[6])


    decile_contribution_plot(
        predictions=predictions,
        true_labels=tst_labels,
        train_labels=trn_labels,
        num_splits=num_splits,
        metric="P",
        k=k,
        title="Dataset",
        dark=False,
        colors=None,
        opacity=[1.0, 0.5],
        fname='decile_plot.png')


if __name__ == "__main__":
    main()