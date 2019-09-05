import numpy as np
from multiprocessing import Pool
import time
from .base import BaseClassifier
from ..utils import shortlist_utils, utils
import logging
from ._svm import train_one
import scipy.sparse as sp
import _pickle as pickle
from functools import partial
import os
from ..data import data_loader
import operator
from ..utils import sparse
from functools import reduce


def sigmoid(X):
    return 1/(1+np.exp(-X))


def separate(result):
    return [item[0] for item in result], [item[1] for item in result]


def convert_to_sparse(weight, bias):
    weight = np.vstack(weight).squeeze()
    bias = np.vstack(bias).squeeze()
    return sp.csr_matrix(weight), sp.csr_matrix(bias).transpose()


class Slice(BaseClassifier):
    def __init__(self, solver='liblinear', loss='squared_hinge', M=100,
                 method='hnsw', efC=300, num_neighbours=300, efS=300,
                 C=1.0, verbose=0, max_iter=20, tol=0.1, threshold=0.01,
                 feature_type='dense', dual=True, use_bias=True,
                 order='centroids', num_threads=12, batch_size=1000,
                 norm='l2'):
        assert feature_type == 'dense', "Not yet tested on sparse features!"
        super().__init__(verbose, use_bias, feature_type)
        self.loss = loss
        self.C = C
        self.verbose = verbose
        self.max_iter = max_iter
        self.threshold = threshold
        self.tol = tol
        self.dual = dual
        self.num_labels = None
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.num_labels = None
        self.valid_labels = None
        self.norm = norm
        self.num_labels_ = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('Slice')
        self.feature_type = feature_type
        self.shorty = None
        self.shorty = shortlist_utils.construct_shortlist(
            method=method, num_neighbours=num_neighbours,
            M=M, efC=efC, efS=efS,
            num_threads=self.num_threads, order=order)
        self.logger.info(self.shorty)
        self.classifiers = None

    def get_data_loader(self, data_dir, dataset, feat_fname,
                        label_fname, mode, batch_order):
        """Data loader
        - batch_order: 'label' during training
        - batch_order: 'instances' during prediction
        """
        return data_loader.DataloaderShortlist(
            batch_size=self.batch_size,
            data_dir=data_dir,
            dataset=dataset,
            feat_fname=feat_fname,
            label_fname=label_fname,
            feature_type=self.feature_type,
            mode=mode,
            batch_order=batch_order,
            norm=self.norm,
            start_index=0,
            end_index=-1)

    def _merge_weights(self, weights, biases):
        # Bias is always a dense array
        if self.feature_type == 'sparse':
            self.weight = sp.vstack(
                weights, format='csr', dtype=np.float32)
            self.bias = sp.vstack(
                biases, format='csr', dtype=np.float32).toarray()
        else:
            self.weight = np.vstack(weights).astype(np.float32).squeeze()
            self.bias = np.vstack(biases).astype(np.float32)

    def fit(self, data_dir, dataset, feat_fname, label_fname,
            model_dir, save_after=1):
        data = self.get_data_loader(
            data_dir, dataset, feat_fname, label_fname, 'train', 'labels')
        self.logger.info("Training Approx. NN!")
        self.shorty.fit(data.features.data, data.labels.data)
        shortlist_indices, shortlist_distance = self.shorty.query(
            data.features.data)
        self.num_labels = data.num_labels  # valid labels
        self.num_labels_ = data.num_labels_  # number of original labels
        self.valid_labels = data.valid_labels
        data.update_data_shortlist(shortlist_indices, shortlist_distance)
        weights, biases = [], []
        run_time = 0.0
        start_time = time.time()
        num_batches = data.num_batches
        for idx, batch_data in enumerate(data):
            start_time = time.time()
            batch_weight, batch_bias = self._train(
                batch_data, self.num_threads)
            del batch_data
            if self.feature_type == 'sparse':
                batch_weight, batch_bias = utils.convert_to_sparse(
                    batch_weight, batch_bias)
            batch_time = time.time() - start_time
            run_time += batch_time
            weights.append(batch_weight), biases.extend(batch_bias)
            self.logger.info(
                "Batch: [{}/{}] completed!, time taken: {}".format(
                    idx+1, num_batches, batch_time))
            if idx != 0 and idx % save_after == 0:
                # TODO: Delete these to save RAM?
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

    def predict(self, data_dir, dataset, feat_fname, label_fname, beta=0.2):
        # TODO Works for batch only; need to loop over all instances otherwise
        data = self.get_data_loader(
            data_dir, dataset, feat_fname, label_fname, 'predict', 'instances')
        num_instances = data.num_instances
        num_features = data.num_features
        predicted_clf = sp.lil_matrix(
            (num_instances, self.num_labels), dtype=np.float32)
        predicted_knn = sp.lil_matrix(
            (num_instances, self.num_labels), dtype=np.float32)
        predicted = sp.lil_matrix(
            (num_instances, self.num_labels), dtype=np.float32)
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
            score_clf = sigmoid(score_clf)
            score_knn = sigmoid(1-np.array(shortlist_dist))
            score = beta*score_clf + (1-beta)*score_knn
            # TODO Works for dense only
            utils._update_predicted_shortlist(
                start_idx, score_clf, predicted_clf,
                np.array(shortlist_indices))
            utils._update_predicted_shortlist(
                start_idx, score_knn, predicted_knn,
                np.array(shortlist_indices))
            utils._update_predicted_shortlist(
                start_idx, score, predicted, np.array(shortlist_indices))
            start_idx += batch_size
            del x_, w_, b_
        end_time = time.time()
        self.logger.info(
            "Prediction time/sample (ms): {}".format(
                (end_time-start_time)*1000/num_instances))
        return predicted_clf, predicted_knn, self._map_to_original(predicted)

    def _map_to_original(self, X):
        """Some labels were removed during training as training data was
        not availale; remap to original mapping
        - Assumes documents need not be remapped
        """
        shape = (X.shape[0], self.num_labels_)
        return sparse._map_cols(X, self.valid_labels, shape)

    def _transpose_weights(self):
        self.weight = self.weight.transpose()
        self.bias = self.bias.transpose()

    def save(self, fname):
        self.shorty.save(fname+".shortlist")
        super().save(fname)

    def load(self, fname):
        self.shorty.load(fname+'.shortlist')
        super().load(fname)

    def __repr__(self):
        return "#Labels: {}, C: {}, Max_iter: {},"
        "Threshold: {}, Loss: {}".format(self.num_labels,
                                         self.C, self.max_iter,
                                         self.threshold, self.loss)
