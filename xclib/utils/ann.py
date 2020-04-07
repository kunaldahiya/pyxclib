"""
    Approximate nearest neighbors with option to perform
    full k-nearest eighbour search or HNSW algorithm
    Use CPUs for computations
    TODO: Add functionanlity to use GPUs
"""

import nmslib
import hnswlib
from sklearn.neighbors import NearestNeighbors
import pickle
import numpy as np


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

    def __init__(self, num_neighbours, method='brute', num_threads=-1):
        self.num_neighbours = num_neighbours
        self.index = NearestNeighbors(
            n_neighbors=num_neighbours, algorithm=method,
            metric='cosine', n_jobs=num_threads
        )

    def fit(self, data):
        self.index.fit(data)

    def predict(self, data):
        distances, indices = self.index.kneighbors(
            X=data, n_neighbors=self.num_neighbours, return_distance=True
        )
        return indices, distances

    def save(self, fname):
        with open(fname, 'wb') as fp:
            pickle.dump({'num_neighbours': self.num_neighbours,
                         'index': self.index}, fp)

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

    def __init__(self, M, efC, efS, num_neighbours, num_threads):
        self.index = nmslib.init(method='hnsw', space='cosinesimil')
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

    def predict(self, data, efS=None):
        efS = efS if efS is not None else self.efS
        self.index.setQueryTimeParams({'efSearch': self.efS})
        output = self.index.knnQueryBatch(
            data, k=self.efS, num_threads=self.num_threads
            )
        indices, distances = self._filter(output)
        return indices, distances

    def save(self, fname):
        nmslib.saveIndex(self.index, fname)

    def load(self, fname):
        nmslib.loadIndex(self.index, fname)


class HNSWM(object):
    def __init__(self, M, efC, efS, num_neighbours, space='ip',
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
        self.index.add_items(data)

    def _filter(self, output):
        indices = []
        distances = []
        for item in output:
            indices.append(item[0])
            distances.append(item[1])
        return indices, distances

    def predict(self, data, efS=None):
        efS = efS if efS is not None else self.efS
        self.index.set_ef(efS)
        indices, distances = self.index.knn_query(data, k=efS)
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
