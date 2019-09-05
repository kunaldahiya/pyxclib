from .ann import NearestNeighbor, HNSW, HNSWM
import logging
from sklearn.preprocessing import normalize
import _pickle as pickle
from .clustering import Cluster
import numpy as np
from collections import OrderedDict
from numba import jit
import time


def construct_shortlist(method, num_neighbours, M, efC, efS,
                        order='centroids', space='cosine',
                        num_threads=-1, num_clusters=1,
                        threshold_freq=10000, verbose=True):
    if order == 'centroids':
        return ShortlistCentroids(
            method, num_neighbours, M, efC, efS, space,
            num_threads, num_clusters, threshold_freq, verbose)
    elif order == 'instances':
        return ShortlistInstances(
            method, num_neighbours, M, efC, efS, space,
            num_threads, verbose)
    else:
        raise NotImplementedError("Unknown order")


class ShortlistBase(object):
    def __init__(self, method, num_neighbours, M, efC, efS, space,
                 num_threads=-1, verbose=False):
        self.method = method
        self.space = space
        self.verbose = verbose
        self.num_neighbours = num_neighbours
        self.M = M
        self.efC = efC
        self.efS = efS
        self.num_threads = num_threads
        self.index = None
        self._construct()

    def _construct(self):
        if self.method == 'brute':
            self.index = NearestNeighbor(
                num_neighbours=self.num_neighbours,
                method='brute',
                num_threads=self.num_threads,
                space=self.space
            )
        elif self.method == 'hnsw':
            self.index = HNSW(
                M=self.M,
                efC=self.efC,
                efS=self.efS,
                num_neighbours=self.num_neighbours,
                num_threads=self.num_threads,
                space=self.space,
                verbose=self.verbose
            )
        elif self.method == 'hnswm':
            self.index = HNSWM(
                M=self.M,
                efC=self.efC,
                efS=self.efS,
                num_neighbours=self.num_neighbours,
                num_threads=self.num_threads,
                space=self.space,
                verbose=self.verbose
            )
        else:
            print("Unknown NN method!")

    def save(self, fname):
        self.index.save(fname)

    def load(self, fname):
        self.index.load(fname)

    def reset(self):
        # TODO Do we need to delete it!
        del self.index
        self._construct()

    def __repr__(self):
        return "Method: {}, efC: {}, efS: {}"
        ", num_neighbors: {}".format(
            self.method, self.efC, self.efS, self.num_neighbours)


class ShortlistInstances(ShortlistBase):
    def __init__(self, method, num_neighbours, M, efC, efS,
                 space, num_threads=-1, verbose=False):
        super().__init__(method, num_neighbours, M, efC, efS,
                         space, num_threads, verbose)
        self.labels = None

    def _compute_similarity(self, distances):
        # Convert distances to similarity
        return list(map(lambda x: 1-x, distances))

    def _remap_one(self, indices, distances):
        similarity = self._compute_similarity(distances)
        out_dict = {}
        for _, (ind, sim) in enumerate(zip(indices, similarity)):
            pos_labels = self.labels[ind]
            for _pl in pos_labels:
                if _pl in out_dict:
                    out_dict[_pl] += sim
                else:
                    out_dict[_pl] = sim
        return list(out_dict.keys()), list(out_dict.values())

    def _remap(self, indices, distances):
        _indices, _distances = [], []
        for _, (ind, dist) in enumerate(zip(indices, distances)):
            _ind, _dist = self._remap_one(ind, dist)
            _indices.append(_ind)
            _distances.append(_dist)
        return _indices, _distances

    def fit(self, features, labels):
        self.index.fit(features)
        self.labels = labels

    def query(self, data):
        indices, distances = self.index.predict(data, self.efS)
        indices, distances = self._remap(indices, distances)
        return indices, distances

    def save(self, fname):
        self.index.save(fname+".index")
        pickle.dump(
            {'labels': self.labels,
             'M': self.M, 'efC': self.efC,
             'efS': self.efS,
             'num_neighbours': self.num_neighbours,
             'space': self.space}, open(fname+".label", 'wb'))

    def load(self, fname):
        self.index.load(fname+".index")
        obj = pickle.load(
            open(fname+".label", 'rb'))
        self.num_neighbours = obj['num_neighbours']
        self.efS = obj['efS']
        self.space = obj['space']
        self.labels = obj['labels'].tolil().rows


class ShortlistCentroids(ShortlistBase):
    def __init__(self, method, num_neighbours, M, efC,
                 efS, space, num_threads=-1, num_clusters=1,
                 threshold_freq=10000, verbose=False):
        super().__init__(
            method, num_neighbours, M, efC,
            efS, space, num_threads, verbose)
        self.num_clusters = num_clusters
        self.threshold_freq = threshold_freq
        self.label_mapping = None
        self.padding_index = None

    def _cluster_instances(self, features, labels, extf_labels):
        """
            Cluster instances for specific labels
        """
        num_features = features.shape[1]
        _clusters = Cluster(
            indices=extf_labels, embedding_dims=num_features,
            num_clusters=self.num_clusters, max_iter=50, n_init=2,
            num_threads=-1)
        _clusters.fit(features, labels)
        return _clusters.predict()

    def _compute_extf_labels_mapping(self, labels):
        """
            Compute extremely frequent labels and get mapping
        """
        def _get_ext_head(labels, threshold):
            freq = np.array(labels.sum(axis=0)).ravel()
            return np.where(freq >= threshold)[0].tolist()
        num_labels = labels.shape[1]
        extf_labels = _get_ext_head(labels, self.threshold_freq)
        # Check if extf_labels is not empty
        if self.num_clusters > 1 and extf_labels:
            self.label_mapping = np.arange(num_labels)
            for idx in extf_labels:
                self.label_mapping = np.append(
                    self.label_mapping, [idx]*self.num_clusters)
        return extf_labels

    def _pad_seq(self, indices, dist):
        _pad_length = self.efS - len(indices)
        indices.extend([self.padding_index]*_pad_length)
        dist.extend([100]*_pad_length)

    def _map_one(self, indices, values, _func=min, _limit=1e5):
        """
            Map indices and values for an instance
        """
        indices = np.asarray(
            list(map(lambda x: self.label_mapping[x], indices)))
        _dict = dict({})
        for id, ind in enumerate(indices):
            _dict[ind] = _func(_dict.get(ind, _limit), values[id])
        indices, values = zip(*_dict.items())
        indices, values = list(indices), list(values)
        if len(indices) < self.efS:
            self._pad_seq(indices, values)
        return indices, values

    def _remap_extf_labels(self, indices, values):
        if self.label_mapping is not None:
            # FIXME Support only batches as of now
            _indices = []
            _values = []
            for _, (index, value) in enumerate(zip(indices, values)):
                _index, _value = self._map_one(index, value)
                _indices.append(_index)
                _values.append(_value)
        else:
            _indices, _values = indices, values
        return _indices, _values

    def fit(self, features, labels):
        """
            Train NN/ANN structure on label centroid
            Also support multiple centroid by clustering labels
            Args:
                features: np.ndarray: document features
                labels: scipy.sparse or None: sparse label matrix
        """
        if labels is None:
            centroids = features  # Fit on given label feature
            self.padding_index = centroids.shape[0]
        else:
            centroids = labels.transpose().dot(features)
            self.padding_index = centroids.shape[0]
            extf_labels = self._compute_extf_labels_mapping(labels)
            # Cluster if required
            if self.label_mapping is not None:
                ext_embeddings = self._cluster_instances(
                    features, labels, extf_labels)
                centroids = np.vstack([centroids, ext_embeddings])
        self.index.fit(centroids)

    def query(self, data):
        indices, distances = self.index.predict(data, self.efS)
        indices, distances = self._remap_extf_labels(indices, distances)
        return indices, distances

    def save(self, fname):
        self.index.save(fname+".index")
        pickle.dump({'label_mapping': self.label_mapping,
                     'num_clusters': self.num_clusters,
                     'threshold_freq': self.threshold_freq,
                     'padding_index': self.padding_index,
                     'M': self.M, 'efC': self.efC, 'efS': self.efS,
                     'space': self.space, 'verbose': self.verbose,
                     'num_neighbours': self.num_neighbours},
                    open(fname+".label", 'wb'))

    def load(self, fname):
        self.index.load(fname+".index")
        obj = pickle.load(open(fname+".label", 'rb'))
        self.label_mapping = obj['label_mapping']
        self.padding_index = obj['padding_index']
        self.num_clusters = obj['num_clusters']
        self.num_neighbours = obj['num_neighbours']
        self.efS = obj['efS']
        self.space = obj['space']
        self.verbose = obj['verbose']
        self.threshold_freq = obj['threshold_freq']

    def __repr__(self):
        return "Method: {}, efC: {}, efS: {},"
        "num_neighbors: {}, num_clusters:{}, " \
            "threshold_freq:{}".format(
                self.method, self.efC, self.efS, self.num_neighbours,
                self.num_clusters, self.threshold_freq)
