import numpy as np
from ..utils import shortlist_utils, utils
from .base import BaseClassifier
import numpy as np
import pickle
import time
import scipy.sparse as sparse
import logging


class MIPS(BaseClassifier):
    """
        MIPS over existing classifiers/label embeddings
    """

    def __init__(self, method, num_neighbours, M=100, efC=300,
                 efS=300, space='ip', num_threads=12, num_clusters=1,
                 verbose=False):
        super().__init__(verbose, False, use_sparse=False)
        self.num_neighbours = num_neighbours
        self.shorty = shortlist_utils.construct_shortlist(
            method=method, num_neighbours=num_neighbours,
            M=M, efC=efC, efS=efS, order='centroids', space=space,
            num_threads=num_threads, num_clusters=num_clusters,
            threshold_freq=None)
        logging.basicConfig(level=logging.INFO)
        self.label_embeddings = None
        self.logger = logging.getLogger('MIPS')

    def fit(self, data, **kwargs):
        """
            Train ANN index over given classifiers/label weights
            Args:
                features: np.ndarray: train set features
                labels: sparse.csr_matrix: sparse label matrix
        """
        self.num_labels = data.num_labels
        self.label_embeddings = data.label_embeddings
        self.shorty.fit(data.label_embeddings, None)

    def predict(self, data, num_neighbours=None, top_k=10):
        """
            Predict using trained classifier
            Args:
                features: np.ndarray: features for test instances
                num_neighbours: int/list: number of neighbors
                batch_size: int: batch size while predicting
                top_k: int: store only top_k predictions 
        """
        # num_samples = data.num_samples
        # predicted = sparse.lil_matrix(
        #     (num_samples, data.num_valid_labels), dtype=np.float32)
        # start_time = time.time()
        # start_idx = 0
        # for _, batch_data in enumerate(data):
        #     pred = batch_data['data'][batch_data['ind']] @ self.label_embeddings.transpose()
        #     gen_utils._update_predicted(
        #         start_idx, pred.toarray() if self.use_sparse else pred,
        #         predicted)
        #     start_idx += pred.shape[0]
        # end_time = time.time()
        # self.logger.info(
        #     "Prediction time/sample (ms): {}".format(
        #         (end_time-start_time)*1000/num_samples))
        # predicted = sparse.csr_matrix(data.features @ self.label_embeddings.transpose())
        if num_neighbours is not None:
            self.shorty.efS = num_neighbours
        predicted = sparse.lil_matrix(
            (data.num_samples, data.num_valid_labels), dtype=np.float32)
        start_time = time.time()
        start_idx = 0
        for _, batch_data in enumerate(data):
            batch_size = len(batch_data['ind'])
            shortlist_indices, shortlist_dist = self.shorty.query(
                batch_data['data'][batch_data['ind']])
            score = utils.sigmoid(-1*np.array(shortlist_dist))
            utils._update_predicted_shortlist(
                start_idx, score, predicted, np.array(shortlist_indices), top_k=top_k)
            start_idx += batch_size
        end_time = time.time()
        self.logger.info(
            "Prediction time/sample (ms): {}".format((end_time-start_time)*1000/data.num_samples))
        return predicted

    def save(self, fname):
        pickle.dump({'use_sparse': self.use_sparse,
                     'num_labels': self.num_labels,
                     'label_embeddings': self.label_embeddings},
                    open(fname, 'wb'))
        self.shorty.save(fname+".shortlist")

    def load(self, fname):
        temp = pickle.load(open(fname, 'rb'))
        self.label_embeddings = temp['label_embeddings']
        self.use_sparse = temp['use_sparse']
        self.num_labels = temp['num_labels']
        self.shorty.load(fname+'.shortlist')

    def __repr__(self):
        return "#Labels: {}, efC: {}, efS: {}, M: {}, num_nbrs: {}".format(self.num_labels,
                                                                           self.shorty.efS, self.shorty.efC,
                                                                           self.shorty.M, self.num_neighbours)
