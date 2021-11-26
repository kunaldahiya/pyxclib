import numpy as np
import time
import logging
from .base import BaseClassifier
import scipy.sparse as sp
from ._svm import train_one
from functools import partial
from ..utils import sparse
from ..data import data_loader
from ._svm import train_one, _get_liblinear_solver_type
from joblib import Parallel, delayed
from ..utils.matrix import SMatrix
from tqdm import tqdm


def separate(result):
    return [item[0] for item in result], [item[1] for item in result]


def convert_to_sparse(weight, bias):
    weight = np.vstack(weight).squeeze()
    bias = np.vstack(bias).squeeze()
    return sp.csr_matrix(weight), sp.csr_matrix(bias).transpose()


class OVAClassifier(BaseClassifier):
    """
    One-vs-all classifier for sparse or dense data
    (suitable for large label set)

    Parameters:
    -----------
    solver: str, optional, default='liblinear'
        solver to use
    loss: str, optional, default='squared_hinge'
        loss to optimize,
        - hinge
        - squared_hinge
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
    num_threads: int, optional, default=10
        use multiple threads to parallelize
    batch_size: int, optional, default=1000
        train these many classifiers in parallel
    norm: str, optional, default='l2'
        normalize features
    penalty: str, optional, default='l2'
        l1 or l2 regularizer
    """

    def __init__(self, solver='liblinear', loss='squared_hinge', C=1.0,
                 verbose=0, max_iter=20, tol=0.1, threshold=0.01,
                 feature_type='sparse', dual=True, use_bias=True,
                 num_threads=12, batch_size=1000, norm='l2', penalty='l2'):
        super().__init__(verbose, use_bias, feature_type)
        self.loss = loss
        self.C = C
        self.penalty = penalty
        self.norm = norm
        self.num_threads = num_threads
        self.verbose = verbose
        self.max_iter = max_iter
        self.threshold = threshold
        self.tol = tol
        self.dual = dual
        self.batch_size = batch_size
        self.num_labels = None
        self.valid_labels = None
        self.num_labels_ = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('OVAClassifier')
        self.logger.info("Parameters:: {}".format(str(self)))

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

    def get_data_loader(self, data_dir, dataset, feat_fname,
                        label_fname, mode, batch_order):
        """Data loader
        - batch_order: 'label' during training
        - batch_order: 'instances' during prediction
        """
        return data_loader.Dataloader(
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
        self.logger.info("Training!")
        data = self.get_data_loader(
            data_dir, dataset, feat_fname, label_fname, 'train', 'labels')
        self.num_labels = data.num_labels  # valid labels
        self.num_labels_ = data.num_labels_  # number of original labels
        self.valid_labels = data.valid_labels
        weights, biases = [], []
        run_time = 0.0
        start_time = time.time()
        idx = 0
        for batch_data in tqdm(data):
            start_time = time.time()
            batch_weight, batch_bias = self._train(
                batch_data, self.num_threads)
            del batch_data
            if self.feature_type == 'sparse':
                batch_weight, batch_bias = convert_to_sparse(
                    batch_weight, batch_bias)
            batch_time = time.time() - start_time
            run_time += batch_time
            weights.append(batch_weight), biases.extend(batch_bias)
            if idx != 0 and idx % save_after == 0:
                #  TODO: Delete these to save memory?
                self._merge_weights(weights, biases)
                self._save_state(model_dir, idx)
                self.logger.info("Saved state at epoch: {}".format(idx))
            idx += 1
        self._merge_weights(weights, biases)
        self.logger.info("Training time (sec): {}, model size (MB): {}".format(
            run_time, self.model_size))

    def _train(self, data, num_threads):
        """Train SVM for multiple labels
        Arguments:
        ---------
        data: list
            [{'X': X, 'Y': y}]
        Returns
        -------
        weights: np.ndarray
            weight of the classifier
        bias: float
            bias of the classifier
        """
        _func = self._get_partial_train()
        with Parallel(n_jobs=num_threads) as parallel:
            result = parallel(delayed(_func)(d) for d in data)
        weights, biases = separate(result)
        del result
        return weights, biases

    def predict(self, data_dir, dataset, feat_fname, label_fname, top_k=10):
        """Predict using the classifier
        Will create batches on instance and then parallelize
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
            TODO: Avoid sending labels as they are not used
        """
        self._transpose_weights()
        self.logger.info("Predicting!")
        use_sparse = self.feature_type == 'sparse'
        data = self.get_data_loader(
            data_dir, dataset, feat_fname, label_fname, 'predict', 'instances')
        num_instances = data.num_instances
        predicted_labels = SMatrix(
            n_rows=num_instances,
            n_cols=self.num_labels,
            nnz=top_k)        
        start_time = time.time()
        start_idx = 0
        for batch_data in tqdm(data):
            pred = batch_data['data'][batch_data['ind']
                                      ] @ self.weight + self.bias
            predicted_labels.update_block(
                start_idx, 
                ind=None,
                val=pred.view(np.ndarray) if use_sparse else pred)
            start_idx += pred.shape[0]
        end_time = time.time()
        self.logger.info(
            "Prediction time/sample (ms): {}".format(
                (end_time-start_time)*1000/num_instances))
        return self._map_to_original(predicted_labels.data())

    def _get_partial_train(self):
        return partial(train_one, solver_type=self.solver, C=self.C,
                       verbose=self.verbose, max_iter=self.max_iter,
                       threshold=self.threshold, tol=self.tol,
                       intercept_scaling=1.0, fit_intercept=self.use_bias,
                       epsilon=0)

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

    def __repr__(self):
        s = "C: {C}, max_iter: {max_iter}, threshold: {threshold}" \
            ", loss: {loss}, dual: {dual}, bias: {use_bias}, norm: {norm}" \
            ", num_threads: {num_threads}, batch_size: {batch_size}"\
            ", tol: {tol}, penalty: {penalty}"
        return s.format(**self.__dict__)

    @property
    def solver(self):
        return _get_liblinear_solver_type(
            'ovr', self.penalty, self.loss, self.dual)
