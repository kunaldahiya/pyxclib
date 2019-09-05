import numpy as np
import os
from ..utils import utils
import sklearn.preprocessing
import scipy.sparse as sparse
import _pickle as pickle
from . import data_utils
from .features import FeaturesBase
from .labels import LabelsBase


class DataloaderBase(object):
    """Base Dataloader for extreme classifiers
    Works for sparse and dense features
    Parameters:
    -----------
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
    batch_size: int, optional, default=1000
        train these many classifiers in parallel
    feature_type: str, optional, default='sparse'
        feature type: sparse or dense
    mode: str, optional, default='train'
        train or predict
        - remove invalid labels in train
    batch_order: str, optional, default='labels'
        iterate over labels or instances
    norm: str, optional, default='l2'
        normalize features
    start_index: int, optional, default=0
        start training from this labels index
    end_index: int, optional, default=-1
        train till this labels index
    """

    def __init__(self, data_dir, dataset, feat_fname, label_fname,
                 batch_size, feature_type, mode='train',
                 batch_order='labels', norm='l2', start_index=0,
                 end_index=-1):
        # TODO Option to load only features; useful in prediction
        self.feature_type = feature_type
        self.norm = norm
        self.features, self.labels = None, None
        self.batch_size = batch_size
        self.start_index = start_index
        self.end_index = end_index
        self.batch_order = batch_order
        self.mode = mode
        self.valid_labels = None
        self.num_valid_labels = None
        self.batches = None
        self.construct(data_dir, dataset, feat_fname, label_fname)

    def load_features(self, data_dir, fname):
        """Load features from given file
        """
        return FeaturesBase(data_dir, fname)

    def load_labels(self, data_dir, fname):
        """Load labels from given file
        Labels can also be supplied directly
        """
        # Pass dummy labels if required
        return LabelsBase(data_dir, fname)

    def load_data(self, data_dir, fname_f, fname_l):
        """Load features and labels from file in libsvm format or pickle
        """
        features = self.load_features(data_dir, fname_f)
        labels = self.load_labels(data_dir, fname_l)
        if self.norm is not None:
            features.normalize(norm=self.norm)
        return features, labels

    @property
    def num_instances(self):
        return self.features.num_instances

    @property
    def num_features(self):
        return self.features.num_features

    @property
    def num_labels(self):
        return self.labels.num_labels

    def get_stats(self):
        """Get dataset statistics
        """
        return self.num_instances, self.num_features, self.num_labels

    def construct(self, data_dir, dataset, feat_fname, label_fname):
        data_dir = os.path.join(data_dir, dataset)
        self.features, self.labels = self.load_data(
            data_dir, feat_fname, label_fname)
        self.num_labels_ = self.labels.Y.shape[1]   # Original number of labels
        if self.mode == 'train':
            if self.start_index != 0 or self.end_index != -1:
                self.end_index = self.num_labels \
                    if self.end_index == -1 else self.end_index
                self.labels = self.labels[:, self.start_index: self.end_index]
            self.valid_labels = self.labels.remove_invalid()
        self._gen_batches()

    def _create_instance_batch(self, batch_indices):
        batch_data = {}
        batch_data['data'] = self.features.data
        batch_data['ind'] = batch_indices
        batch_data['Y'] = self.labels.index_select(batch_indices, axis=0)
        return batch_data

    def _gen_batches(self):
        if self.batch_order == 'labels':
            offset = 0 if self.num_labels % self.batch_size == 0 else 1
            num_batches = self.num_labels//self.batch_size + offset
            self.batches = np.array_split(
                np.arange(self.num_labels), num_batches)
        elif self.batch_order == 'instances':
            offset = 0 if self.num_instances % self.batch_size == 0 else 1
            num_batches = self.num_instances//self.batch_size + offset
            self.batches = np.array_split(
                np.arange(self.num_instances), num_batches)
        else:
            raise NotImplementedError("Unknown order for batching!")

    def save(self, fname):
        state = {'num_labels': self.num_labels,
                 'num_labels_': self.num_labels_,
                 'valid_labels': self.valid_labels
                 }
        pickle.dump(state, open(fname, 'wb'))

    def load(self, fname):
        state = pickle.load(open(fname, 'rb'))
        self.num_labels = state['num_labels']
        self.num_labels_ = state['num_labels_']
        self.valid_labels = state['valid_labels']

    @property
    def num_batches(self):
        return len(self.batches)  # Number of batches


class Dataloader(DataloaderBase):
    """Base Dataloader for 1-vs-all extreme classifiers
    Works for sparse and dense features
    Parameters:
    -----------
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
    batch_size: int, optional, default=1000
        train these many classifiers in parallel
    feature_type: str, optional, default='sparse'
        feature type: sparse or dense
    mode: str, optional, default='train'
        train or predict
        - remove invalid labels in train
    batch_order: str, optional, default='labels'
        iterate over labels or instances
    norm: str, optional, default='l2'
        normalize features
    start_index: int, optional, default=0
        start training from this labels index
    end_index: int, optional, default=-1
        train till this labels index
    """

    def __init__(self, data_dir, dataset, feat_fname, label_fname,
                 batch_size, feature_type, mode='train',
                 batch_order='labels', norm='l2', start_index=0,
                 end_index=-1):
        super().__init__(data_dir, dataset, feat_fname, label_fname,
                         batch_size, feature_type, mode, batch_order, norm,
                         start_index, end_index)

    def _create_label_batch(self, batch_indices):
        batch_data = []
        for idx in batch_indices:
            item = {}
            pos_indices = self.labels.index_select(idx).indices.tolist()
            batch_labels = -1*np.ones((self.num_instances,), dtype=np.float32)
            batch_labels[pos_indices] = 1
            item['ind'] = None  # SVM won't slice data
            item['data'] = self.features.data
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
    """Base Dataloader for 1-vs-all extreme classifiers
    Works for sparse and dense features
    Parameters:
    -----------
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
    batch_size: int, optional, default=1000
        train these many classifiers in parallel
    feature_type: str, optional, default='sparse'
        feature type: sparse or dense
    mode: str, optional, default='train'
        train or predict
        - remove invalid labels in train
    batch_order: str, optional, default='labels'
        iterate over labels or instances
    norm: str, optional, default='l2'
        normalize features
    start_index: int, optional, default=0
        start training from this labels index
    end_index: int, optional, default=-1
        train till this labels index
    """

    def __init__(self, data_dir, dataset, feat_fname, label_fname,
                 batch_size, feature_type, mode='train',
                 batch_order='labels', norm='l2', start_index=0,
                 end_index=-1):
        # TODO Option to load only features; useful in prediction
        super().__init__(data_dir, dataset, feat_fname, label_fname,
                         batch_size, feature_type, mode, batch_order, norm,
                         start_index, end_index)

    def _create_label_batch(self, batch_indices):
        batch_data = []
        for idx in batch_indices:
            item = {}
            item['data'] = self.features.data
            #  TODO Check if this could be done more efficiently
            temp = self.labels.index_select(idx)
            item['ind'] = temp.indices
            item['Y'] = temp.data
            batch_data.append(item)
        return batch_data

    def _create_batch(self, batch_indices):
        if self.batch_order == 'labels':
            return self._create_label_batch(batch_indices)
        else:
            return self._create_instance_batch(batch_indices)

    def update_data_shortlist(self, shortlist_indices, shortlist_dist):
        # TODO Remove this loop
        _labels = self.labels.data.tolil()  # Avoid this?
        for idx in range(self.num_instances):
            pos_labels = _labels[idx, :].__dict__['rows'][0]
            neg_labels = list(
                filter(lambda x: x not in set(pos_labels),
                       shortlist_indices[idx]))
            _labels[idx, neg_labels] = -1
        self.labels.data = _labels

    def __iter__(self):
        for _, batch_indices in enumerate(self.batches):
            batch_data = self._create_batch(batch_indices)
            yield batch_data
