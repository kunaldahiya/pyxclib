import numpy as np
import os
from . import data_utils
from ..utils import utils
import sklearn.preprocessing
import scipy.sparse as sparse
import _pickle as pickle
import time


class DataloaderBase(object):
    """
        Dataloader object for 1-vs-all classifiers
        Works for sparse and dense features
        Params:
            features: csr_matrix or np.ndarray
            labels: csr_matrix: (num_samples, num_labels)
            batch_size: int: label batch size
            use_sparse: bool: use sparse features or dense
            use_shortlist: train using all negatives or shortlist
    """

    def __init__(self, batch_size, use_sparse, mode='train',
                 batch_order='labels', normalize_=True, start_index=0, 
                 end_index=-1):
        # TODO Option to load only features; useful in prediction
        self.use_sparse = use_sparse
        self.normalize_ = normalize_
        self.features, self.labels = None, None
        self.batch_size = batch_size
        self.num_samples, self.num_features = None, None
        self.num_labels = None
        self.start_index = start_index
        self.end_index = end_index
        self.batch_order = batch_order
        self.num_valid_labels = None
        self.mode = mode
        self.valid_labels = None
        self.num_valid_labels = None
        self.labels_ = None
        self.batches = None

    def construct(self, data_fname):
        self.features, self.labels, self.num_samples, self.num_features, \
            self.num_labels = self._load_data(data_fname)
        self.num_labels = self.num_labels
        if self.mode == 'train':
            if self.start_index != 0 or self.end_index != -1:
                self.end_index = self.num_labels if self.end_index == -1 else self.end_index
                self.labels = self.labels[:, self.start_index: self.end_index]
                self.num_labels = self.labels.shape[1]
            self.valid_labels = utils.get_valid_labels(self.labels)
        self.num_valid_labels = self.valid_labels.size
        self.labels_ = self.labels[:, self.valid_labels]
        self.features_ = self.features
        self._gen_batches()

    def _load_data(self, fname):
        if '.pkl' in fname:
            temp = pickle.load(open(fname, 'rb'))
            features, labels = temp['features'], temp['labels']
            num_samples, num_features = features.shape
            num_labels = labels.shape[1]
        else:
            features, labels, num_samples, num_features, num_labels = data_utils.read_data(fname)
            labels = data_utils.binarize_labels(labels, num_labels)
            if not self.use_sparse:
                features = features.toarray()
        if self.normalize_:
            features = sklearn.preprocessing.normalize(features, copy=False)
        return features.astype(np.float32), labels.tocsc().astype(np.float32),\
            num_samples, num_features, num_labels

    def _create_instance_batch(self, batch_indices):
        batch_data = {}
        batch_data['data'] = self.features
        batch_data['ind'] = batch_indices
        # batch_data['X'] = self.features[batch_indices, :]
        batch_data['Y'] = self.labels_[batch_indices, :]
        return batch_data

    def _gen_batches(self):
        if self.batch_order == 'labels':
            offset = 0 if self.num_valid_labels % self.batch_size == 0 else 1
            num_batches = self.num_valid_labels//self.batch_size + offset
            self.batches = np.array_split(
                np.arange(self.num_valid_labels), num_batches)
        elif self.batch_order == 'instances':
            offset = 0 if self.num_samples % self.batch_size == 0 else 1
            num_batches = self.num_samples//self.batch_size + offset
            self.batches = np.array_split(
                np.arange(self.num_samples), num_batches)
        else:
            raise NotImplementedError("Unknown order for batching!")

    def save(self, fname):
        state = {'num_labels': self.num_labels,
                 'valid_labels': self.valid_labels
                 }
        pickle.dump(state, open(fname, 'wb'))

    def load(self, fname):
        state = pickle.load(open(fname, 'rb'))
        self.num_labels = state['num_labels']
        self.valid_labels = state['valid_labels']

    def _num_batches(self):
        return (self.batches) #Number of batches