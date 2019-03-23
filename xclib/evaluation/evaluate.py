# Example to evaluate
import sys
import xclibs.evaluation.xml_metrics
import xclibs.data.data_utils
from scipy.io import loadmat
import numpy as np

def main(targets_file, train_file, predictions_file):
    # Load the dataset
    _, te_labels, te_num_samples, _, te_num_labels = data_utils.read_data(targets_file)
    true_labels = data_utils.binarize_labels(te_labels, te_num_labels)
    
    _, tr_labels, tr_num_samples, _, tr_num_labels = data_utils.read_data(train_file)
    trn_labels  = data_utils.binarize_labels(tr_labels, tr_num_labels)

    predicted_labels = loadmat(predictions_file)['predicted_labels']
    inv_propen = xml_metrics.compute_inv_propesity(trn_labels, A= 0.55, B=1.5)
    acc = xml_metrics.Metrices(true_labels, inv_propensity_scores=inv_propen, remove_invalid=False)
    args = acc.eval(predicted_labels, 5)
    print(xml_metrics.format(*args))

if __name__ == '__main__':
    train_file = sys.argv[1]
    targets_file = sys.argv[2] # Usually test data file
    predictions_file = sys.argv[3] # In mat format
    main(targets_file, train_file, predictions_file)
