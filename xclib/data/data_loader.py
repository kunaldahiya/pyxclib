import numpy as np
import os
from xctools.data import data_utils
from .data_loader_base import DataloaderBase
from ..utils import utils
import sklearn.preprocessing
import scipy.sparse as sparse
import _pickle as pickle
import time


class Dataloader(DataloaderBase):
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
        super().__init__(batch_size, use_sparse, mode, batch_order, normalize_,
                         start_index, end_index)

    def _create_label_batch(self, batch_indices):
        batch_data = []
        for idx in batch_indices:
            item = {}
            pos_indices = self.labels_[:, idx].indices.tolist()
            batch_labels = -1*np.ones((self.num_samples,), dtype=np.float32)
            batch_labels[pos_indices] = 1
            item['ind'] = None  # SVM won't slice data
            item['data'] = self.features
            item['Y'] = batch_labels  # +1/-1 for SVM
            batch_data.append(item)
        return batch_data

    def _create_batch(self, batch_indices):
        if self.batch_order == 'labels':
            return self._create_label_batch(batch_indices)
        else:
            return self._create_instance_batch(batch_indices)

    def __iter__(self):
        for _, batch_indices in enumerate(self.batches):
            batch_data = self._create_batch(batch_indices)
            yield batch_data


class DataloaderShortlist(DataloaderBase):
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
                 batch_order='labels', normalize_=True, 
                 start_index=0, end_index=-1):
        # TODO Option to load only features; useful in prediction
        super().__init__(batch_size, use_sparse, mode, batch_order, normalize_,
                    start_index, end_index)

    def _create_label_batch(self, batch_indices):
        batch_data = []
        for idx in batch_indices:
            item = {}
            item['data'] = self.features
            #  TODO Check if this could be done more efficiently
            temp = self.labels_[:, idx] 
            item['ind'] = temp.__dict__['indices']
            item['Y'] = temp.__dict__['data']
            batch_data.append(item)
        return batch_data

    def _create_batch(self, batch_indices):
        if self.batch_order == 'labels':
            return self._create_label_batch(batch_indices)
        else:
            return self._create_instance_batch(batch_indices)

    def update_data_shortlist(self, shortlist_indices, shortlist_dist):
        # TODO Remove this loop
        self.labels_ = self.labels_.tolil() # Avoid this?
        for idx in range(self.num_samples):
            pos_labels = self.labels_[idx, :].__dict__['rows'][0]
            neg_labels = list(filter(lambda x: x not in set(pos_labels), shortlist_indices[idx]))
            self.labels_[idx, neg_labels] = -1
        self.labels_ = self.labels_.tocsc()

    def __iter__(self):
        for _, batch_indices in enumerate(self.batches):
            batch_data = self._create_batch(batch_indices)
            yield batch_data
