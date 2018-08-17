# Example to evaluate
import sys
sys.path.append('../data')
import xml_metrics
import data_utils
from scipy.io import loadmat

def main(targets_file, predictions_file):
    # Load the dataset
    _, labels, num_samples, _, num_labels = data_utils.read_data(targets_file)
    true_labels = data_utils.binarize_labels(labels, num_labels)
    predicted_labels = loadmat(predictions_file)['predicted_labels']
    acc = xml_metrics.Metrices(true_labels, predicted_labels)
    print(acc.compute_all(5))

if __name__ == '__main__':
    targets_file = sys.argv[1] # Usually test data file
    predictions_file = sys.argv[2] # In mat format
    main(targets_file, predictions_file)
