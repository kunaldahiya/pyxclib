# Approximate nearest neighbors with option to perform full k-nearest eighbour search
# Use CPUs for computations
# TODO: Add functionanlity to use GPUs

import nmslib
import hnswlib
from sklearn.neighbors import NearestNeighbors
import pickle


class NearestNeighbor(object):
    def __init__(self, num_neighbours, method='brute', space='cosine', num_threads=-1):
        self.num_neighbours = num_neighbours
        self.index = NearestNeighbors(n_neighbors=num_neighbours, algorithm=method, metric=space, n_jobs=num_threads)

    def fit(self, data):
        self.index.fit(data)

    def predict(self, data, num_neighbours=None):
        num_neighbours = num_neighbours if num_neighbours is not None else self.num_neighbours
        distances, indices = self.index.kneighbors(X=data, n_neighbors=num_neighbours, return_distance=True)
        return indices, distances

    def save(self, fname):
        with open(fname, 'wb') as fp:
            pickle.dump({'num_neighbours': self.num_neighbours, 
                         'index': self.index}, fp
                        )

    def load(self, fname):
        with open(fname, 'rb') as fp:
            temp = pickle.load(fp)
            self.index = temp['index']
            self.num_neighbours = temp['num_neighbours']


class HNSW(object):
    def __init__(self, M, efC, efS, num_neighbours, space='cosine', num_threads=12, verbose=True):
        if space == 'cosine':
            space = 'cosinesimil'
        self.verbose = verbose
        self.index = nmslib.init(method='hnsw', space=space)     
        self.M = M
        self.num_threads = num_threads
        self.efC = efC
        self.efS = efS
        self.num_neighbours = num_neighbours

    def fit(self, data):
        self.index.addDataPointBatch(data)
        self.index.createIndex({'M': self.M,
                                'indexThreadQty': self.num_threads,
                                'efConstruction': self.efC},
                               print_progress=self.verbose
                               )

    def _filter(self, output):
        indices = []
        distances = []
        for item in output:
            indices.append(item[0])
            distances.append(item[1])
        return indices, distances
     
    def predict(self, data, efS=None):
        efS = efS if efS is not None else self.efS
        self.index.setQueryTimeParams({'efSearch': efS})
        output = self.index.knnQueryBatch(data, k=efS,
                                          num_threads=self.num_threads, 
                                        )
        indices, distances = self._filter(output)
        return indices, distances

    def save(self, fname):
        nmslib.saveIndex(self.index, fname)

    def load(self, fname):
        nmslib.loadIndex(self.index, fname)


class HNSWM(object):
    def __init__(self, M, efC, efS, num_neighbours, space='ip', num_threads=12, verbose=False):
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
        self.index.init_index(max_elements=self.max_elements, ef_construction=self.efC, M=self.M)
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
        
