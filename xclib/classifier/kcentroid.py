import numpy as np
from ..utils import shortlist_utils, utils
from .base import BaseClassifier
import operator
import _pickle as pickle
import time
import scipy.sparse as sparse
import logging


class KCentroidClassifier(BaseClassifier):
    """
        K-Centroid classifier
    """

    def __init__(self, method, num_neighbours, M=100, efC=300, efS=300, num_threads=12,
                 num_clusters=1, threshold_freq=10000, verbose=False, use_sparse=False):
        super().__init__(verbose, False, use_sparse)
        self.num_neighbours = num_neighbours
        self.shorty = shortlist_utils.construct_shortlist(
            method=method, num_neighbours=num_neighbours,
            M=M, efC=efC, efS=efS, order='centroids',
            num_threads=num_threads, num_clusters=num_clusters,
            threshold_freq=threshold_freq)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('KCentroid')

    def fit(self, data, **kwargs):
        """
            Train Approx. nearest centroid classifier
            Args:
                features: np.ndarray: train set features
                labels: sparse.csr_matrix: sparse label matrix
        """
        self.num_labels = data.num_labels
        start_time = time.time()
        self.shorty.fit(data.features, data.labels)
        end_time = time.time()
        self.logger.info(
            "Train time (s): {}".format((end_time-start_time)))

    def predict(self, data, num_neighbours=None, top_k=10):
        """
            Predict using trained classifier
            Args:
                features: np.ndarray: features for test instances
                num_neighbours: int/list: number of neighbors
                batch_size: int: batch size while predicting
                top_k: int: store only top_k predictions 
        """
        if num_neighbours is not None:
            self.shorty.efS = num_neighbours
        predicted = sparse.lil_matrix(
            (data.num_samples, data.num_valid_labels+1), dtype=np.float32)
        start_time = time.time()
        start_idx = 0
        for _, batch_data in enumerate(data):
            batch_size = len(batch_data['ind'])
            shortlist_indices, shortlist_dist = self.shorty.query(
                batch_data['data'][batch_data['ind']])
            score = utils.sigmoid(1-np.array(shortlist_dist))
            utils._update_predicted_shortlist(
                start_idx, score, predicted, np.array(shortlist_indices), top_k=top_k)
            start_idx += batch_size
        end_time = time.time()
        self.logger.info(
            "Prediction time/sample (ms): {}".format((end_time-start_time)*1000/data.num_samples))
        return predicted[:, :-1]

    def save(self, fname):
        pickle.dump({'use_sparse': self.use_sparse,
                     'num_labels': self.num_labels},
                    open(fname, 'wb'))
        self.shorty.save(fname+".shortlist")

    def load(self, fname):
        temp = pickle.load(open(fname, 'rb'))
        self.use_sparse = temp['use_sparse']
        self.num_labels = temp['num_labels']
        self.shorty.load(fname+'.shortlist')

    def __repr__(self):
        return "#Labels: {}, efC: {}, efS: {}, M: {}, num_nbrs: {}".format(self.num_labels,
                                                                           self.shorty.efS, self.shorty.efC,
                                                                           self.shorty.M, self.num_neighbours)
