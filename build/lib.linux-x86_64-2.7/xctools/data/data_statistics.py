"""
Compute statistics for XC datasets.
"""

from scipy.sparse import csr_matrix, vstack
import numpy as np 
import json

class Statistics(object):
    """
        n_train_samples: Number of train samples
        n_test_samples: Number of train samples
        n_features: Number of features
        n_labels: Number of labels
        n_avg_samples_per_label: Number of average samples per label
        n_avg_labels_per_sample: Number of average labels per sample
        avg_doc_length: Average document length
    """
    def __init__(self):
        self.n_train_samples = None
        self.n_test_samples = None
        self.n_features = None
        self.n_labels = None
        self.n_avg_samples_per_label = None
        self.n_avg_labels_per_sample = None
        self.avg_doc_length = None

    def compute_avg_samples_per_label(self, labels):
        return labels.sum(axis=0).mean()

    def compute_avg_labels_per_sample(self, labels):
        return labels.sum(axis=1).mean()

    def compute_avg_doc_length(self, features):
        return features.astype(np.bool).sum(axis=1).mean()

    def compute(self, train_features, train_labels, test_features=None, test_labels=None):
        """
            Compute features for given data. Test data is optional.
            Args:
                train_features: csr_matrix: train features
                train_labels: csr_matrix: train labels
                test_features: csr_matrix: test features
                test_labels: csr_matrix: test labels
        """
        if test_features is not None:
            self.n_test_samples = test_features.shape[0]
            features = vstack([train_features, test_features]).tocsr()
            labels = vstack([train_labels, test_labels]).tocsr()
        else:
            self.n_train_samples, self.n_features = train_features.shape
            self.n_labels = train_labels.shape[1] 
            features = train_features
            labels = train_labels
        self.n_avg_samples_per_label = self.compute_avg_samples_per_label(labels)
        self.n_avg_labels_per_sample = self.compute_avg_labels_per_sample(labels)
        self.avg_doc_length = self.compute_avg_doc_length(features)

    def write(self, fname):
        """
            Write statistics to a given file in json format
        """
        with open(fname, 'w') as outfile:
            json.dump(vars(self), outfile, indent=4)
