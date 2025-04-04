"""
    Approximate nearest neighbors with option to perform
    full k-nearest eighbour search or HNSW algorithm
    Use CPUs for computations
    TODO: Add functionanlity to use GPUs
"""
from numpy import ndarray

import pickle
import numpy as np
import nmslib
import hnswlib
from .dense import topk, normalize
from sklearn.neighbors import NearestNeighbors
from .clustering import cluster_balance, b_kmeans_dense
try:
    import torch
except ImportError:
    print("Pytorch is not installed; GPU based search won't work!")


class NearestNeighborGPU(object):
    """Nearest Neighbor in brute-force fashion
    * computation is done on GPU
    """
    def __init__(
            self, 
            num_neighbours: int, 
            fp16: bool = False, 
            sorted: bool = False,
            append_bias: bool = False, 
            space: str = 'cosine',
            *args, **kwargs):
        """
        Args:
            num_neighbours (int): number of neighbours to search
            method (int, optional): method for knn search 
                Defaults to 'brute-gpu'.
            batch_size (int, optional): batch size for search. Defaults to 256.
            fp16 (bool, optional): use half precision or not. Defaults to False.
            sorted (bool, optional): sort the predictions. Defaults to False.
            space (str, optional): search space. Defaults to 'cosine'.
        """
        self.num_neighbours = num_neighbours
        self.index = None
        self.fp16 = fp16
        self.sorted = sorted
        self.space = space
        self.append_bias = append_bias
        self.device = "cuda"

    def fit(self, data: ndarray) -> None:
        if self.space == 'cosine':
            data = torch.from_numpy(normalize(data))
        else:
            data = torch.from_numpy(data)
        if self.fp16:
            self.index = data.half().T.to(self.device)
        else:
            self.index = data.T.to(self.device)

    def _predict(self, data: ndarray) -> tuple[ndarray, ndarray]:
        if self.space == 'cosine':
            data = torch.from_numpy(normalize(data))
        else:
            data = torch.from_numpy(data)
        if self.append_bias:
            data = torch.hstack(
                [data, torch.ones((len(data), 1), dtype=data.dtype)])
        scores = data.to(self.device) @ self.index
        val, ind = torch.topk(scores, k=self.num_neighbours)
        return ind.cpu().numpy(), val.cpu().numpy()

    def _set_query_time_params(self, num_neighbours: int = None):
        if num_neighbours is not None:
            self.num_neighbours = num_neighbours

    def predict(
            self, 
            data: ndarray,
            num_neighbours: int = None) -> tuple[ndarray, ndarray]:
        """Get closest vectors

        Args:
            data (ndarray): input data
            num_neighbours (int, optional): get these many neighbors.
                Defaults to None.

        Returns:
            tuple[ndarray, ndarray]: indices and values of top neighbors
        """
        self._set_query_time_params(num_neighbours)
        return self._predict(data)

    def query(
            self, 
            data: ndarray,
            num_neighbours: int = None) -> tuple[ndarray, ndarray]:
        """Get closest vectors

        Args:
            data (ndarray): input data
            num_neighbours (int, optional): get these many neighbors.
                Defaults to None.

        Returns:
            tuple[ndarray, ndarray]: indices and values of top neighbors
        """
        return self.predict(data, num_neighbours)

    def save(self, fname: str) -> None:
        with open(fname, 'wb') as fp:
            pickle.dump({'num_neighbours': self.num_neighbours,
                         'index': self.index,
                         'fp16': self.fp16,
                         'sorted': self.sorted,
                         'space': self.space},
                        fp, protocol=4)

    def load(self, fname: str) -> None:
        with open(fname, 'rb') as fp:
            temp = pickle.load(fp)
            self.index = temp['index']
            self.num_neighbours = temp['num_neighbours']
            self.fp16 = temp['fp16']
            self.sorted = temp['sorted']
            self.space = temp['space']


class NearestNeighbor(object):
    """Nearest Neighbor using knn algorithm
    Parameters
    ----------
    num_neighbours: int
        number of neighbours to search
    method: str, optional, default='brute'
        method for knn search brute/ball_tree/kd_tree
    num_threads: int: optional, default=-1
        #threads to cluster
    """

    def __init__(self, num_neighbours, method='brute', num_threads=-1,
                 space='cosine'):
        self.num_neighbours = num_neighbours
        self.index = NearestNeighbors(
            n_neighbors=num_neighbours, algorithm=method,
            metric=space, n_jobs=num_threads
        )

    def fit(self, data):
        self.index.fit(data)

    def _predict(self, data):
        distances, indices = self.index.kneighbors(
            X=data, n_neighbors=self.num_neighbours, return_distance=True
        )
        return indices, distances

    def _set_query_time_params(self, num_neighbours=None):
        if num_neighbours is not None:
            self.num_neighbours = num_neighbours

    def predict(self, data, num_neighbours=None):
        self._set_query_time_params(num_neighbours)
        return self._predict(data)

    def save(self, fname):
        with open(fname, 'wb') as fp:
            pickle.dump({'num_neighbours': self.num_neighbours,
                         'index': self.index}, fp, protocol=4)

    def load(self, fname):
        with open(fname, 'rb') as fp:
            temp = pickle.load(fp)
            self.index = temp['index']
            self.num_neighbours = temp['num_neighbours']


class HNSW(object):
    """HNSW algorithm for ANN Search
    Parameters
    ----------
    M: int
        HNSW M (Usually 100)
    efC: int
        construction parameter (Usually 300)
    efS: int
        search parameter (Usually 300)
    num_neighbours: int
        #neighbours
    num_threads: int
        #threads to cluster
    """

    def __init__(self, M, efC, efS, num_neighbours, num_threads,
                 space='cosine'):
        space_map = {'cosine': 'cosinesimil'}
        space = space_map[space]
        self.index = nmslib.init(method='hnsw', space=space)
        self.M = M
        self.num_threads = num_threads
        self.efC = efC
        self.efS = efS
        self.num_neighbours = num_neighbours

    def fit(self, data, print_progress=True):
        self.index.addDataPointBatch(data)
        self.index.createIndex(
            {'M': self.M,
             'indexThreadQty': self.num_threads,
             'efConstruction': self.efC},
            print_progress=print_progress
            )

    def _filter(self, output):
        indices = np.zeros((len(output), self.efS), dtype=np.int64)
        distances = np.zeros((len(output), self.efS), dtype=np.float32)
        for idx, item in enumerate(output):
            indices[idx] = item[0]
            distances[idx] = item[1]
        return indices, distances

    def _predict(self, data):
        output = self.index.knnQueryBatch(
            data, k=self.num_neighbours, num_threads=self.num_threads)
        indices, distances = self._filter(output)
        return indices, distances

    def _set_query_time_params(self, efS=None, num_neighbours=None):
        self.efS = efS if efS is not None else self.efS
        if num_neighbours is not None:
            self.num_neighbours = num_neighbours
        self.index.setQueryTimeParams({'efSearch': self.efS})

    def predict(self, data, efS=None, num_neighbours=None):
        self._set_query_time_params(efS, num_neighbours)
        indices, distances = self._predict(data)
        return indices, distances

    def save(self, fname):
        nmslib.saveIndex(self.index, fname)

    def load(self, fname):
        nmslib.loadIndex(self.index, fname)


class HNSWLib(object):
    def __init__(self, M, efC, efS, num_neighbours, space='cosine',
                 num_threads=12, verbose=False):
        self.verbose = verbose
        self.index = None
        self.M = M
        self.num_threads = num_threads
        self.efC = efC
        self.efS = efS
        self.dim = None
        self.space = space
        self.max_elements = None
        self.num_neighbours = num_neighbours

    def _init(self):
        self.index = hnswlib.Index(space=self.space, dim=self.dim)

    def fit(self, data):
        self.max_elements, self.dim = data.shape
        self._init()
        self.index.init_index(
            max_elements=self.max_elements, ef_construction=self.efC, M=self.M)
        self.index.add_items(data, num_threads=self.num_threads)

    def _set_query_time_params(self, efS=None, num_neighbours=None):
        self.efS = efS if efS is not None else self.efS
        if num_neighbours is not None:
            self.num_neighbours = num_neighbours
        self.index.set_ef(self.efS)

    def _predict(self, data):
        return self.index.knn_query(
            data, k=self.num_neighbours, num_threads=self.num_threads)

    def predict(self, data, efS=None, num_neighbours=None):
        self._set_query_time_params(efS, num_neighbours)
        indices, distances = self._predict(data)
        return indices, distances

    def save(self, fname):   
        with open(fname+".params", 'wb') as fp:
            pickle.dump({'dim': self.dim,
                         'max_elements': self.max_elements}, fp
                        )
        self.index.save_index(fname)

    def load(self, fname):
        with open(fname+".params", 'rb') as fp:
            obj = pickle.load(fp)
            self.dim = obj['dim']
            self.max_elements = obj['max_elements']
        self._init()
        self.index.load_index(fname)


class ClusteringIndex(object):
    """Clustering Index

    Caution: It will always return num_neighbours items when quering
    So, set efS carefully, otherwise it'll pad with num_items 
    """
    def __init__(
            self,
            num_clusters,
            efS,
            num_neighbours,
            num_threads=6,
            space='cosine'):
        self.num_clusters = num_clusters
        self.pad_ind = -1
        self.pad_val = -1
        self.num_threads = num_threads
        self.dim = None
        self.index = None
        self.efS = efS
        self.num_neighbours = num_neighbours
        self.avg_size = None
        self.cluster_reps = None
        self.space = space
        self._init()

    def _init(self):
        pass

    def fit(self, X, num_clusters=None):
        _nc = self.num_clusters if num_clusters is None else num_clusters

        self.dim = X.shape[1]
        self.cluster_reps = np.zeros((self.num_clusters, self.dim), dtype='float32')

        self.index, _ = cluster_balance(
            X=X.astype('float32'), 
            clusters=[np.arange(len(X), dtype='int')],
            num_clusters=_nc,
            splitter=b_kmeans_dense,
            num_threads=self.num_threads,
            verbose=True)
        for i, ind in enumerate(self.index):
            self.cluster_reps[i] = np.mean(X[ind], axis=0)
        self.cluster_reps = self.cluster_reps.T
        self.avg_size = np.mean(list(map(len, self.index)))
        self.pad_ind = len(X)

    def query(self, x):
        c_sims = x @ self.cluster_reps 
        n = len(x)
        c_ind, c_sims = topk(
            c_sims,
            k=self.efS,
            sorted=True)

        ind = np.full((n, self.num_neighbours), self.pad_ind, dtype='int64')
        val = np.full((n, self.num_neighbours), self.pad_val, dtype='float32')

        for i in range(n):
            s = 0
            for ci, cs  in zip(c_ind[i], c_sims[i]):
                e = min(s + len(self.index[ci]), self.num_neighbours)
                ind[i, s: e] = self.index[ci][:e-s]
                val[i, s: e] = cs
                if e == self.num_neighbours:
                    break
                else:
                    s = e
        return ind, val

    def save(self, fname):
        with open(fname, 'wb') as fp:
            pickle.dump({'index': self.index,
                         'cluster_reps': self.cluster_reps}, fp
                        )

    def load(self, fname):
        with open(fname, 'rb') as fp:
            obj = pickle.load(fp)
            self.index = obj['index']
            self.cluster_reps = obj['cluster_reps']
