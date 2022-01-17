from .ann import NearestNeighbor, HNSW, HNSWLib
import pickle
from .clustering import Cluster
import numpy as np
import numba as nb
from ..utils.dense import compute_centroid
import os
import math


@nb.njit(cache=True)
def bin_index(array, item): # Binary search
    first, last = 0, len(array) - 1

    while first <= last:
        mid = (first + last) // 2
        if array[mid] == item:
            return mid

        if item < array[mid]:
            last = mid - 1
        else:
            first = mid + 1

    return -1


@nb.njit(cache=True)
def safe_normalize(array):
    _max = np.max(array)
    if _max != 0:
        return array/_max
    else:
        return array


@nb.njit(nb.types.Tuple(
    (nb.int64[:], nb.float32[:]))(nb.int64[:, :], nb.float32[:], nb.int64))
def map_one(indices_labels, similarity, pad_ind):
    unique_point_labels = np.unique(indices_labels)
    unique_point_labels = unique_point_labels[unique_point_labels != pad_ind]
    point_label_similarity = np.zeros(
        (len(unique_point_labels), ), dtype=np.float32)
    for j in range(len(indices_labels)):
        for lbl in indices_labels[j]:
            if(lbl != pad_ind):
                _ind = bin_index(unique_point_labels, lbl)
                point_label_similarity[_ind] += similarity[j]
    point_label_similarity = safe_normalize(point_label_similarity)
    return unique_point_labels, point_label_similarity


@nb.njit(nb.types.Tuple(
    (nb.int64[:, :], nb.float32[:, :]))(nb.int64[:, :], nb.float32[:, :],
     nb.int64[:, :], nb.int64, nb.int64, nb.float32), parallel=True)
def map_neighbors(indices, similarity, labels, top_k, pad_ind, pad_val):
    m = indices.shape[0]
    point_labels = np.full(
        (m, top_k), pad_ind, dtype=np.int64)
    point_label_sims = np.full(
        (m, top_k), pad_val, dtype=np.float32)
    for i in nb.prange(m):
        unique_point_labels, point_label_sim = map_one(
            labels[indices[i]], similarity[i], pad_ind)
        if top_k < len(unique_point_labels):
            top_indices = np.argsort(
                point_label_sim)[-1 * top_k:][::-1]
            point_labels[i] = unique_point_labels[top_indices]
            point_label_sims[i] = point_label_sim[top_indices]
        else:
            point_labels[i, :len(unique_point_labels)] = unique_point_labels
            point_label_sims[i, :len(unique_point_labels)] = point_label_sim
    return point_labels, point_label_sims


@nb.njit(cache=True)
def _remap_centroid_one(indices, sims, mapping):
    mapped_indices = mapping[indices]
    unique_mapped_indices = np.unique(mapped_indices)
    unique_mapped_sims = np.zeros(
        (len(unique_mapped_indices), ), dtype=np.float32)
    for i in range(len(unique_mapped_indices)):
        ind = unique_mapped_indices[i]
        unique_mapped_sims[i] = np.max(sims[mapped_indices == ind])
    return unique_mapped_indices, unique_mapped_sims


@nb.njit()
def map_centroids(indices, sims, mapping, pad_ind, pad_val):
    mapped_indices = np.full(
        indices.shape, fill_value=pad_ind, dtype=np.int64)
    mapped_sims = np.full(
        indices.shape, fill_value=pad_val, dtype=np.float32)

    for i in nb.prange(indices.shape[0]):
        _ind, _sim = _remap_centroid_one(indices[i], sims[i], mapping)
        mapped_indices[i, :len(_ind)] = _ind
        mapped_sims[i, :len(_sim)] = _sim
    return mapped_indices, mapped_sims


def construct_shortlist(method, num_neighbours, M, efC, efS,
                        order='centroids', space='cosine',
                        num_threads=-1, num_clusters=1,
                        threshold_freq=10000, verbose=True):
    if order == 'centroids':
        return ShortlistCentroids(
            method, num_neighbours, M, efC, efS, num_threads,
            space, verbose, num_clusters, threshold_freq)
    elif order == 'instances':
        return ShortlistInstances(
            method, num_neighbours, M, efC, efS, num_threads,
            space, verbose)
    else:
        raise NotImplementedError("Unknown order")


class Shortlist(object):
    """Get nearest neighbors using brute or HNSW algorithm
    Parameters
    ----------
    method: str
        brute or hnsw
    num_neighbours: int
        number of neighbors
    M: int
        HNSW M (Usually 100)
    efC: int
        construction parameter (Usually 300)
    efS: int
        search parameter (Usually 300)
    num_threads: int, optional, default=24
        use multiple threads to build index
    space: str, optional (default='cosine')
        search in this space 'cosine', 'ip'
    """

    def __init__(self, method, num_neighbours, M, efC, 
                 efS, num_threads=24, space='cosine'):
        self.method = method
        self.num_neighbours = num_neighbours
        self.M = M
        self.efC = efC
        self.space = space
        self.efS = efS
        self.num_threads = num_threads
        self.index = None
        self._construct()

    def _construct(self):
        if self.method == 'brute':
            self.index = NearestNeighbor(
                num_neighbours=self.num_neighbours,
                method='brute',
                num_threads=self.num_threads
            )
        elif self.method == 'hnswlib':
            self.index = HNSWLib(
                space=self.space,
                M=self.M,
                efC=self.efC,
                efS=self.efS,
                num_neighbours=self.num_neighbours,
                num_threads=self.num_threads
            )
        elif self.method == 'hnsw':
            self.index = HNSW(
                space=self.space,
                M=self.M,
                efC=self.efC,
                efS=self.efS,
                num_neighbours=self.num_neighbours,
                num_threads=self.num_threads
            )
        else:
            print("Unknown NN method!")

    def fit(self, data):
        self.index.fit(data)

    def query(self, data, *args, **kwargs):
        indices, distances = self.index.predict(data, *args, **kwargs)
        return indices, 1-distances

    def save(self, fname):
        self.index.save(fname)

    def load(self, fname):
        self.index.load(fname)

    def reset(self):
        # TODO Do we need to delete it!
        del self.index
        self._construct()

    @property
    def model_size(self, fname=None):
        # size on disk; see if there is a better solution
        if fname is None:
            import tempfile
            with tempfile.NamedTemporaryFile() as tmp:
                self.index.save(tmp.name)
                _size = os.path.getsize(tmp.name)/math.pow(2, 20)
        else: #useful when can't create file in tmp
            self.index.save(fname)
            _size = os.path.getsize(fname)/math.pow(2, 20)
            os.remove(fname)
        return _size

    def __repr__(self):
        return "efC: {}, efS: {}, M: {}, num_nbrs: {}, num_threads: {}".format(
            self.efS, self.efC, self.M, self.num_neighbours, self.num_threads)


class ShortlistCentroids(Shortlist):
    """Get nearest labels using KCentroids
    * centroid(l) = mean_{i=1}^{N}{x_i*y_il}
    * brute or HNSW algorithm for search
    Parameters
    ----------
    method: str, optional, default='hnsw'
        brute or hnsw
    num_neighbours: int
        number of neighbors (same as efS)
        * may be useful if the NN search retrieve less number of labels
        * typically doesn't happen with HNSW etc.
    M: int, optional, default=100
        HNSW M (Usually 100)
    efC: int, optional, default=300
        construction parameter (Usually 300)
    efS: int, optional, default=300
        search parameter (Usually 300)
    num_threads: int, optional, default=18
        use multiple threads to cluster
    space: str, optional, default='cosine'
        metric to use while quering
    verbose: boolean, optional, default=True
        print progress
    num_clusters: int, optional, default=1
        cluster instances => multiple representatives for chosen labels
    threshold: int, optional, default=5000
        cluster instances if a label appear in more than 'threshold'
        training points
    """
    def __init__(self, method='hnsw', num_neighbours=300, M=100, efC=300,
                 efS=300, num_threads=24, space='cosine', verbose=True,
                 num_clusters=1, threshold=7500, pad_val=-10000):
        super().__init__(
            method, num_neighbours, M, efC, efS, num_threads, space)
        self.num_clusters = num_clusters
        self.space = space
        self.pad_ind = -1
        self.mapping = None
        self.ext_head = None
        self.threshold = threshold
        self.pad_val = pad_val

    def _cluster_multiple_rep(self, features, labels, label_centroids,
                              multi_centroid_indices):
        embedding_dims = features.shape[1]
        _cluster_obj = Cluster(
            indices=multi_centroid_indices,
            embedding_dims=embedding_dims,
            num_clusters=self.num_clusters,
            max_iter=50, n_init=2, num_threads=-1)
        _cluster_obj.fit(features, labels)
        label_centroids = np.vstack(
            [label_centroids, _cluster_obj.predict()])
        return label_centroids

    def process_multiple_rep(self, features, labels, label_centroids):
        freq = np.array(labels.sum(axis=0)).ravel()
        if np.max(freq) > self.threshold and self.num_clusters > 1:
            self.ext_head = np.where(freq >= self.threshold)[0]
            print("Found {} super-head labels".format(len(self.ext_head)))
            self.mapping = np.arange(label_centroids.shape[0])
            for idx in self.ext_head:
                self.mapping = np.append(
                    self.mapping, [idx]*self.num_clusters)
            return self._cluster_multiple_rep(
                features, labels, label_centroids, self.ext_head)
        else:
            return label_centroids

    def fit(self, features, labels, *args, **kwargs):
        self.pad_ind = labels.shape[1]
        label_centroids = compute_centroid(features, labels, reduction='mean')
        label_centroids = self.process_multiple_rep(
            features, labels, label_centroids)
        super().fit(label_centroids)

    def query(self, data, *args, **kwargs):
        indices, sim = super().query(data, *args, **kwargs)
        return self._remap(indices, sim)

    def _remap(self, indices, sims):
        if self.mapping is None:
            return indices, sims
        return map_centroids(
            indices, sims, self.mapping, self.pad_ind, self.pad_val)

    def load(self, fname):
        temp = pickle.load(open(fname+".metadata", 'rb'))
        self.pad_ind = temp['pad_ind']
        self.pad_val = temp['pad_val']
        self.mapping = temp['mapping']
        self.ext_head = temp['ext_head']
        super().load(fname+".index")

    def save(self, fname):
        metadata = {
            'pad_ind': self.pad_ind,
            'pad_val': self.pad_val,
            'mapping': self.mapping,
            'ext_head': self.ext_head
        }
        pickle.dump(metadata, open(fname+".metadata", 'wb'))
        super().save(fname+".index")

    def purge(self, fname):
        # purge files from disk
        if os.path.isfile(fname+".index"):
            os.remove(fname+".index")
        if os.path.isfile(fname+".metadata"):
            os.remove(fname+".metadata")

    def __repr__(self):
        s = "efC: {efC}, efS: {efS}, M: {M}, num_nbrs: {num_neighbours}" \
            ", pad_ind: {pad_ind}, num_threads: {num_threads}" \
            ", pad_val: {pad_val}, threshold: {threshold}" \
            ", num_clusters: {num_clusters}"
        return s.format(**self.__dict__)


class ShortlistInstances(Shortlist):
    """Get nearest labels using KNN
    * brute or HNSW algorithm for search
    Parameters
    ----------
    method: str, optional, default='hnsw'
        brute or hnsw
    num_neighbours: int
        number of labels to keep per data point
        * labels may be shared across fetched instances
        * union of labels can be large when dataset is densly tagged
    M: int, optional, default=100
        HNSW M (Usually 100)
    efC: int, optional, default=300
        construction parameter (Usually 300)
    efS: int, optional, default=300
        search parameter (Usually 300)
    num_threads: int, optional, default=18
        use multiple threads to cluster
    space: str, optional, default='cosine'
        metric to use while quering
    verbose: boolean, optional, default=True
        print progress
    pad_val: int, optional, default=-10000
        value for padding indices
        - Useful as all documents may have different number of nearest
        labels after collasping them
    """
    def __init__(self, method='hnsw', num_neighbours=300, M=100, efC=300,
                 efS=300, num_threads=24, space='cosine', verbose=False,
                 pad_val=-10000):
        super().__init__(method, efS, M, efC, efS, num_threads, space)
        self.labels = None
        self._num_neighbours = num_neighbours
        self.space = space
        self.pad_ind = None
        self.pad_val = pad_val

    def _remove_invalid(self, features, labels):
        # Keep data points with nnz features and atleast one label
        ind_ft = np.where(np.sum(np.square(features), axis=1) > 0)[0]
        ind_lb = np.where(np.sum(labels, axis=1) > 0)[0]
        ind = np.intersect1d(ind_ft, ind_lb)
        return features[ind], labels[ind]

    def _as_array(self, labels):
        n_pos_labels = list(map(len, labels))
        _labels = np.full(
            (len(labels), max(n_pos_labels)),
            self.pad_ind, np.int64)
        for ind, _lab in enumerate(labels):
            _labels[ind, :n_pos_labels[ind]] = labels[ind]
        return _labels

    def _remap(self, indices, distances):
        return map_neighbors(
            indices, 1-distances,
            self.labels, self._num_neighbours,
            self.pad_ind, self.pad_val)

    def fit(self, features, labels):
        features, labels = self._remove_invalid(features, labels)
        self.index.fit(features)
        self.pad_ind = labels.shape[1]
        self.labels = self._as_array(labels.tolil().rows)

    def query(self, data, *args, **kwargs):
        indices, distances = self.index.predict(data)
        indices = indices.astype(np.int64)
        indices, similarities = self._remap(indices, distances)
        return indices, similarities

    def save(self, fname):
        self.index.save(fname+".index")
        pickle.dump(
            {'labels': self.labels,
             'M': self.M, 'efC': self.efC,
             'efS': self.efS,
             'pad_ind': self.pad_ind,
             'pad_val': self.pad_val,
             'num_neighbours': self._num_neighbours,
             'space': self.space}, 
             open(fname+".metadata", 'wb'),
             protocol=4)

    def load(self, fname):
        self.index.load(fname+".index")
        obj = pickle.load(
            open(fname+".metadata", 'rb'))
        self._num_neighbours = obj['num_neighbours']
        self.efS = obj['efS']
        self.space = obj['space']
        self.labels = obj['labels']
        self.pad_ind = obj['pad_ind']
        self.pad_val = obj['pad_val']

    def purge(self, fname):
        # purge files from disk
        if os.path.isfile(fname+".index"):
            os.remove(fname+".index")
        if os.path.isfile(fname+".metadata"):
            os.remove(fname+".metadata")

    def __repr__(self):
        s = "efC: {efC}, efS: {efS}, M: {M}, num_nbrs: {_num_neighbours}" \
            ", pad_ind: {pad_ind}, num_threads: {num_threads}" \
            ", pad_val: {pad_val}"
        return s.format(**self.__dict__)
