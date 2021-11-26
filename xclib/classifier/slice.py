import numpy as np
from joblib import Parallel, delayed
import time
from .base import BaseClassifier
from ..utils import shortlist, sparse
import logging
from ._svm import train_one, _get_liblinear_solver_type
import scipy.sparse as sp
from functools import partial
from ..data import data_loader
from ..utils.matrix import SMatrix
import time
from tqdm import tqdm


def sigmoid(X):
    return 1/(1+np.exp(-X))


def separate(result):
    return [item[0] for item in result], [item[1] for item in result]


def convert_to_sparse(weight, bias):
    weight = np.vstack(weight).squeeze()
    bias = np.vstack(bias).squeeze()
    return sp.csr_matrix(weight), sp.csr_matrix(bias).transpose()


class Slice(BaseClassifier):
    """
    Slice classifier for dense data
    (suitable for large label set)

    Parameters:
    -----------
    solver: str, optional, default='liblinear'
        solver to use
    loss: str, optional, default='squared_hinge'
        loss to optimize,
        - hinge
        - squared_hinge
    penalty: str, optional, default='l2'
        l1 or l2 regularizer
    M: int, optional, default=100
        HNSW parameter
    efC: int, optional, default=300
        HNSW construction parameter
    efS: int, optional, default=e00
        HNSW search parameter
    num_neighbours: int, optional, default=e00
        use num_neighbours labels for each data point
    C: float, optional, default=1.0
        cost in svm
    verbose: int, optional, default=0
        print progress in svm
    max_iter: int, optional, default=20
        iteration in solver
    tol: float, optional, default=0.1
        tolerance in solver
    threshold: float, optional, default=0.01
        threshold for hard thresholding (after training classifier)
        - bias values are not touched
        - 0.01: for sparse features
        - 1e-5: for dense features
    feature_type: str, optional, default='sparse'
        feature type: sparse or dense
    dual: boolean, optional, default=true
        solve in primal or dual
    use_bias: boolean, optional, default=True
        train bias parameter or not
    order: str, optional, default='centroids'
        create shortlist from KCentroid and KNN
    num_threads: int, optional, default=10
        use multiple threads to parallelize
    batch_size: int, optional, default=1000
        train these many classifiers in parallel
    norm: str, optional, default='l2'
        normalize features
    penalty: str, optional, default='l2'
        l1 or l2 regularizer
    beta: float, optional, default=0.2
        weight of classifier component
    """

    def __init__(self, solver='liblinear', loss='squared_hinge', M=100,
                 method='hnswlib', efC=300, num_neighbours=300, efS=300,
                 C=1.0, verbose=0, max_iter=20, tol=0.001, threshold=0.01,
                 feature_type='dense', dual=True, use_bias=True,
                 order='centroids', num_threads=12, batch_size=1000,
                 norm='l2', penalty='l2', beta=0.2):
        assert feature_type == 'dense', "Not yet tested on sparse features!"
        super().__init__(verbose, use_bias, feature_type)
        self.loss = loss
        self.C = C
        self.penalty = penalty
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
        self.beta = beta
        self.num_labels_ = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('Slice')
        self.feature_type = feature_type
        self.shorty = None
        self.shorty = shortlist.construct_shortlist(
            method=method, num_neighbours=num_neighbours,
            M=M, efC=efC, efS=efS,
            num_threads=self.num_threads,
            order=order)
        self.classifiers = None
        self.logger.info("Parameters:: {}".format(str(self)))

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
        self.weight = np.vstack(weights).astype(np.float32).squeeze()
        self.bias = np.vstack(biases).astype(np.float32)

    def fit(self, data_dir, dataset, feat_fname, label_fname,
            model_dir, save_after=1):
        """Train the classifier
        Will create batches on labels and then parallelize
        - Not very efficient when training time per classifier is too low
        - Will not train for labels without any datapoints
          A list will be maintained which will used to remap labels
          to original ids
        Arguments:
        ---------
        data_dir: str
            data directory with all files
        dataset: str
            Name of the dataset; like EURLex-4K
        feat_fname: str
            File name of training feature file
            Should be in sparse format with header
        label_fname: str
            File name of training label file
            Should be in sparse format with header
        model_dir: str
            dump checkpoints in this directory
            based on save_after
        save_after: int, default=1
            save checkpoints after these many steps
        """
        run_time = 0.0
        data = self.get_data_loader(
            data_dir, dataset, feat_fname, label_fname, 'train', 'labels')
        self.logger.info("Training Approx. NN!")
        start_time = time.time()
        self.shorty.fit(data.features.data, data.labels.data)
        shortlist_indices, shortlist_sim = self.shorty.query(
            data.features.data)
        self.num_labels = data.num_labels  # valid labels
        self.num_labels_ = data.num_labels_  # number of original labels
        self.valid_labels = data.valid_labels
        data.update_data_shortlist(shortlist_indices, shortlist_sim)
        run_time = time.time() - start_time
        weights, biases = [], []
        start_time = time.time()
        self.logger.info("Training classifiers!")
        idx = 0
        for batch_data in tqdm(data):
            start_time = time.time()
            batch_weight, batch_bias = self._train(
                batch_data, self.num_threads)
            del batch_data
            batch_time = time.time() - start_time
            run_time += batch_time
            weights.append(batch_weight), biases.extend(batch_bias)
            if idx != 0 and idx % save_after == 0:
                # TODO: Delete these to save RAM?
                self._merge_weights(weights, biases)
                self._save_state(model_dir, idx)
                self.logger.info("Saved state at epoch: {}".format(idx))
            idx += 1
        self._merge_weights(weights, biases)
        self.logger.info("Training time (sec): {}, model size (MB): {}".format(
            run_time, self.model_size))

    def _train(self, data, num_threads):
        """
        Train SVM for multiple labels
        Args:
            data: list: [{'X': X, 'Y': y}]
        Returns:
            weights: np.ndarray: (num_labels, feature_dims) 
            bias: np.ndarray: (num_labels, 1)
        """
        _func = self._get_partial_train()
        with Parallel(n_jobs=num_threads) as parallel:
            result = parallel(delayed(_func)(d) for d in data)
        weights, biases = separate(result)
        del result
        return weights, biases

    def predict(self, data_dir, dataset, feat_fname, label_fname,
                beta=None, top_k=10):
        if beta is not None:
            self.beta = beta
        # TODO Works for batch only; need to loop over all instances otherwise
        # Append padding index
        self.weight = np.vstack(
            [self.weight, np.zeros((1, self.weight.shape[1]), dtype='float32')])
        self.bias = np.vstack(
            [self.bias, np.full((1, 1), fill_value=-1e5, dtype='float32')])
        data = self.get_data_loader(
            data_dir, dataset, feat_fname, label_fname, 'predict', 'instances')
        num_instances = data.num_instances
        num_features = data.num_features
        predicted = SMatrix(
            n_rows=num_instances,
            n_cols=self.num_labels,
            nnz=top_k)        
        start_time = time.time()
        start_idx = 0
        # This is required so that it doesn't set/print
        # info regarding query params repeatedly
        self.shorty.index._set_query_time_params()
        for batch_data in tqdm(data):
            batch_size = len(batch_data['ind'])
            temp_data = batch_data['data'][batch_data['ind']]

            # Need to do this; otheriwse it prints stuff from set query
            shortlist_indices, shortlist_dist = self.shorty.index._predict(
                temp_data)
            shortlist_sim = 1 - shortlist_dist 
            # Shortlist may contain pad labels
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
            score_knn = sigmoid(shortlist_sim)
            score = self.beta*score_clf + (1-self.beta)*score_knn
            predicted.update_block(
                start_idx, 
                ind=shortlist_indices,
                val=score)
            start_idx += batch_size
            del x_, w_, b_
        end_time = time.time()
        self.logger.info(
            "Prediction time/sample (ms): {}".format(
                (end_time-start_time)*1000/num_instances))
        return self._map_to_original(predicted.data()[:, :-1])

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
        self.shorty.save(fname)
        super().save(fname)

    def load(self, fname):
        self.shorty.load(fname)
        super().load(fname)

    def _get_partial_train(self):
        return partial(train_one, solver_type=self.solver, C=self.C,
                       verbose=self.verbose, max_iter=self.max_iter,
                       threshold=self.threshold, tol=self.tol,
                       intercept_scaling=1.0, fit_intercept=self.use_bias,
                       epsilon=0)

    def __repr__(self):
        s = "{shorty}, C: {C}, max_iter: {max_iter}, threshold: {threshold}" \
            ", loss: {loss}, dual: {dual}, bias: {use_bias}, norm: {norm}" \
            ", tol: {tol}, num_threads: {num_threads}" \
            ", batch_size: {batch_size}"
        return s.format(**self.__dict__)

    @property
    def model_size(self):
        return self.shorty.model_size + super().model_size

    @property
    def solver(self):
        return _get_liblinear_solver_type(
            'ovr', self.penalty, self.loss, self.dual)
