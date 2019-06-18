import numpy as np
from ..utils import shortlist_utils, utils
from .base import BaseClassifier
import logging
import operator
import time
import _pickle as pickle
import scipy.sparse as sparse


class KNeighborsClassifier(BaseClassifier):
    """
        K-neighbor classifier
    """

    def __init__(self, method, num_neighbours, M=100, efC=300, efS=300, num_threads=12,
                 verbose=False, use_sparse=False):
        super().__init__(verbose, False, use_sparse)
        self.num_neighbours = num_neighbours
        self.shorty = shortlist_utils.construct_shortlist(
            method=method, num_neighbours=num_neighbours,
            M=M, efC=efC, efS=efS, order='instances',
            num_threads=num_threads)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('KNN')

    def fit(self, data, **kwargs):
        """
            Train Approx. nearest neighbor classifier
            Args:
                features: np.ndarray: train set features
                labels: sparse.csr_matrix: sparse label matrix
        """
        self.num_labels = data.num_labels
        self.shorty.fit(data.features, data.labels)

    def _predict_one(self, start_index, indices, similarity, predicted):
        # TODO: Avoid this loop with padding?
        for i, (_index, _similarity) in enumerate(zip(indices, similarity)):
            predicted['rows'].extend([i+start_index]*len(_index))
            predicted['cols'].extend(_index)
            predicted['vals'].extend(_similarity)

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
        predicted = {'rows': [], 'cols': [], 'vals': []}
        start_time = time.time()
        start_idx = 0
        for batch_idx, batch_data in enumerate(data):
            batch_size = len(batch_data['ind'])
            shortlist_indices, shortlist_sim = self.shorty.query(
                batch_data['data'][batch_data['ind']])
            self._predict_one(start_idx, shortlist_indices,
                              shortlist_sim, predicted)
            start_idx += batch_size
            self.logger.info(
                "Batch: {} completed!".format(batch_idx))
        end_time = time.time()
        self.logger.info(
            "Prediction time/sample (ms): {}".format((end_time-start_time)*1000/data.num_samples))
        return sparse.csr_matrix((predicted['vals'], (predicted['rows'], predicted['cols'])),
                                 shape=(data.num_samples, data.num_valid_labels))

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
