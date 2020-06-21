import sys
import xclib.evaluation.xc_metrics as xc_metrics
import xclib.data.data_utils as data_utils


def compute_inv_propensity(train_file, A, B):
    """
        Compute Inverse propensity values
        Values for A/B:
            Wikpedia-500K: 0.5/0.4
            Amazon-670K, Amazon-3M: 0.6/2.6
            Others: 0.55/1.5
    """
    train_labels = data_utils.read_sparse_file(train_file)
    inv_propen = xc_metrics.compute_inv_propesity(train_labels, A, B)
    return inv_propen


def main(targets_file, train_file, predictions_file, A, B):
    """
        Args:
            targets_file: test labels
            train_file: train labels (to compute prop)
            prediction_file: predicted labels
            A: int: to compute propensity
            B: int: to compute propensity
    """
    true_labels = data_utils.read_sparse_file(targets_file)
    predicted_labels = data_utils.read_sparse_file(predictions_file)
    inv_psp = compute_inv_propensity(train_file, A, B)
    acc = xc_metrics.Metrics(true_labels=true_labels,
                             inv_psp=inv_psp)
    args = acc.eval(predicted_labels, 5)
    print(xc_metrics.format(*args))


if __name__ == '__main__':
    train_file = sys.argv[1]
    targets_file = sys.argv[2]
    predictions_file = sys.argv[3]
    A = float(sys.argv[4])
    B = float(sys.argv[5])
    main(targets_file, train_file, predictions_file, A, B)
