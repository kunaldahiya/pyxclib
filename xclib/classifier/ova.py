import numpy as np
from multiprocessing import Pool
import time
import logging
from .base import BaseClassifier
import scipy.sparse as sparse
from ._svm import train_one
import _pickle as pickle
from functools import partial
from ..utils import utils
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


class OVAClassifier(BaseClassifier):
    def __init__(self, solver='liblinear', loss='squared_hinge', C=1.0,
                 verbose=0, max_iter=20, tol=0.1, threshold=0.01,
                 use_sparse=True, dual=True, use_bias=True, num_threads=12):
        super().__init__(verbose, use_bias, use_sparse)
        self.loss = loss
        self.C = C
        self.num_threads = num_threads
        self.verbose = verbose
        self.max_iter = max_iter
        self.threshold = threshold
        self.tol = tol
        self.dual = dual
        self.num_labels = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('OVAClassifier')

    def _merge_weights(self, weights, biases):
        """
            Transposed classifiers before saving.
        """
        # Bias is always a dense array
        if self.use_sparse:
            self.weight = sparse.vstack(
                weights, format='csr', dtype=np.float32)
            self.bias = sparse.vstack(biases, format='csr', dtype=np.float32).toarray()
        else:
            self.weight = np.vstack(weights).astype(np.float32).squeeze()
            self.bias = np.vstack(biases).astype(np.float32)

    def fit(self, data, model_dir, save_after=1):
        self.logger.info("Training!")
        self.num_labels = data.num_labels
        weights, biases = [], []
        run_time = 0.0
        start_time = time.time()
        for idx, batch_data in enumerate(data):
            start_time = time.time()
            batch_weight, batch_bias = self._train(
                batch_data, self.num_threads)
            del batch_data
            if self.use_sparse:
                batch_weight, batch_bias = convert_to_sparse(
                    batch_weight, batch_bias)
            batch_time = time.time() - start_time
            run_time += batch_time
            weights.append(batch_weight), biases.extend(batch_bias)
            self.logger.info(
                "Batch: {} completed!, time taken: {}".format(idx, batch_time))
            if idx != 0 and idx % save_after == 0:
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

    def predict(self, data):
        num_samples = data.num_samples
        predicted_labels = sparse.lil_matrix(
            (num_samples, data.num_valid_labels), dtype=np.float32)
        start_time = time.time()
        start_idx = 0
        for _, batch_data in enumerate(data):
            pred = batch_data['data'][batch_data['ind']
                                      ] @ self.weight + self.bias
            utils._update_predicted(
                start_idx, pred.view(np.ndarray) if self.use_sparse else pred,
                predicted_labels)
            start_idx += pred.shape[0]
        end_time = time.time()
        self.logger.info(
            "Prediction time/sample (ms): {}".format(
                (end_time-start_time)*1000/num_samples))
        return predicted_labels

    def __repr__(self):
        return "#Labels: {}, C: {}, Max_iter: {}, Threshold: {}, "\
            "Loss: {}".format(self.num_labels,
                              self.C, self.max_iter,
                              self.threshold, self.loss)
