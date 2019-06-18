import numpy as np
from multiprocessing import Pool
import time
from .base import BaseClassifier
from ..utils import shortlist_utils, utils
import logging
from ._svm import train_one
import scipy.sparse as sparse
import _pickle as pickle
from functools import partial
import os
import _pickle as pickle
import operator
from functools import reduce
from scipy.sparse import csr_matrix


def separate(result):
    return [item[0] for item in result], [item[1] for item in result]


def convert_to_sparse(weight, bias):
    weight = np.vstack(weight).squeeze()
    bias = np.vstack(bias).squeeze()
    return csr_matrix(weight), csr_matrix(bias).transpose()


class Slice(BaseClassifier):
    def __init__(self, solver='liblinear', loss='squared_hinge', M=100, method='hnsw',
                 efC=300, num_neighbours=300, efS=300, C=1.0, verbose=0, max_iter=20,
                 tol=0.1, threshold=0.01, use_sparse=True, dual=True, use_bias=True,
                 order='centroids', num_threads=12):
        super().__init__(verbose, use_bias, use_sparse)
        self.loss = loss
        self.C = C
        self.verbose = verbose
        self.max_iter = max_iter
        self.threshold = threshold
        self.tol = tol
        self.dual = dual
        self.num_labels = None
        self.num_threads = num_threads
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('Slice')
        self.use_sparse = use_sparse
        self.shorty = None
        self.shorty = shortlist_utils.construct_shortlist(
            method=method, num_neighbours=num_neighbours,
            M=M, efC=efC, efS=efS,
            num_threads=self.num_threads, order=order)
        self.logger.info(self.shorty)
        self.classifiers = None

    def _merge_weights(self, weights, biases):
        """
            Transposed classifiers before saving.
        """
        if self.use_sparse:
            self.weight = sparse.vstack(
                weights, format='csr', dtype=np.float32)
            self.bias = sparse.vstack(biases, format='csr', dtype=np.float32)
        else:
            self.weight = np.vstack(weights).squeeze()
            self.bias = np.vstack(biases)

    def fit(self, data, model_dir, save_after=1):
        self.logger.info("Training Approx. NN!")
        self.shorty.fit(data.features, data.labels_)
        shortlist_indices, shortlist_distance = self.shorty.query(
            data.features)
        data.update_data_shortlist(shortlist_indices, shortlist_distance)
        weights, biases = [], []
        run_time = 0.0
        start_time = time.time()
        num_batches = data._num_batches()
        for idx, batch_data in enumerate(data):
            start_time = time.time()
            batch_weight, batch_bias = self._train(batch_data, self.num_threads)
            del batch_data
            if self.use_sparse:
                batch_weight, batch_bias = utils.convert_to_sparse(
                    batch_weight, batch_bias)
            batch_time = time.time() - start_time
            run_time += batch_time
            weights.append(batch_weight), biases.extend(batch_bias)
            self.logger.info(
                "Batch: [{}/{}] completed!, time taken: {}".format(idx+1, num_batches, batch_time))
            if idx != 0 and idx % save_after == 0:
                #TODO: Delete these to save RAM?
                self._merge_weights(weights, biases)
                self._save_state(model_dir, idx)
                self.logger.info("Saved state at epoch: {}".format(idx))
        self._merge_weights(weights, biases)
        self.logger.info("Training time (sec): {}, model size (GB): {}".format(
            run_time, self._compute_clf_size()))

    def _train(self, data, num_threads):
        """
            Train SVM for multiple labels
            Args:
                data: list: [{'X': X, 'Y': y}]
            Returns:
                params_: np.ndarray: (num_labels, feature_dims+1) 
                                    +1 for bias; bias is last term
        """
        with Pool(num_threads) as p:
            _func = partial(train_one, loss=self.loss,
                            C=self.C, verbose=self.verbose,
                            max_iter=self.max_iter, tol=self.tol,
                            threshold=self.threshold, dual=self.dual)
            result = p.map(_func, data)
        weights, biases = separate(result)
        del result
        return weights, biases

    def predict(self, data, beta):
        # TODO Works for batch only; need to loop over all instances otherwise
        self.weight = self.weight.transpose()
        self.bias = self.bias.transpose()
        num_samples = data.num_samples
        num_features = data.num_features
        predicted_clf = sparse.lil_matrix(
            (num_samples, data.num_valid_labels), dtype=np.float32)
        predicted_knn = sparse.lil_matrix(
            (num_samples, data.num_valid_labels), dtype=np.float32)
        predicted = sparse.lil_matrix(
            (num_samples, data.num_valid_labels), dtype=np.float32)
        start_time = time.time()
        start_idx = 0
        for _, batch_data in enumerate(data):
            batch_size = len(batch_data['ind'])
            temp_data = batch_data['data'][batch_data['ind']]
            shortlist_indices, shortlist_dist = self.shorty.query(
                temp_data)
            shortlist_indices_fl = []
            for item in shortlist_indices:
                shortlist_indices_fl.extend(item)
            x_ = temp_data[:, np.newaxis, :]
            w_ = self.weight[shortlist_indices_fl].reshape(
                batch_size, -1, num_features).swapaxes(1, 2)
            b_ = self.bias[shortlist_indices_fl].reshape(
                batch_size, -1)
            score_clf = np.matmul(x_, w_).squeeze() + b_
            score_clf = utils.sigmoid(score_clf)
            score_knn = utils.sigmoid(1-np.array(shortlist_dist))
            score = beta*score_clf + (1-beta)*score_knn
            # TODO Works for dense only
            utils._update_predicted_shortlist(
                start_idx, score_clf, predicted_clf, np.array(shortlist_indices))
            utils._update_predicted_shortlist(
                start_idx, score_knn, predicted_knn, np.array(shortlist_indices))
            utils._update_predicted_shortlist(
                start_idx, score, predicted, np.array(shortlist_indices))
            start_idx += batch_size
            del x_, w_, b_
        end_time = time.time()
        self.logger.info(
            "Prediction time/sample (ms): {}".format((end_time-start_time)*1000/num_samples))
        return predicted_clf, predicted_knn, predicted

    def save(self, fname):
        pickle.dump({'weight': self.weight.transpose(),
                     'bias': self.bias.transpose(),
                     'use_sparse': self.use_sparse,
                     'num_labels': self.num_labels},
                    open(fname, 'wb'))
        self.shorty.save(fname+".shortlist")

    def load(self, fname):
        temp = pickle.load(open(fname, 'rb'))
        self.bias = temp['bias']
        self.weight = temp['weight']
        self.use_sparse = temp['use_sparse']
        self.num_labels = temp['num_labels']
        self.shorty.load(fname+'.shortlist')

    def __repr__(self):
        return "#Labels: {}, C: {}, Max_iter: {}, Threshold: {}, Loss: {}".format(self.num_labels,
                                                                                  self.C, self.max_iter,
                                                                                  self.threshold, self.loss)
