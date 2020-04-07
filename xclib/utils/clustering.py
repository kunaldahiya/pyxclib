# Use CPUs for computations

from sklearn.cluster import KMeans
import pickle
import numpy as np


class Cluster(object):
    """Cluster given data for given label indices in given number
    of clusters
    Parameters
    ----------
    indices: [int]
        Cluster these labels
    embedding_dims: int
        dimensionality of the space
    num_clusters: int, optional, default=300
        number of cluster in each set
    max_iter: int, optional, default=50
        #iterations in KMeans Algorithm
    n_init: int, optional, default=2
        #initializations in KMeans Algorithm
    num_threads: int, optional, default=-1
        number of threads for each job
    """

    def __init__(self, indices, embedding_dims, num_clusters=300,
                 max_iter=50, n_init=2, num_threads=-1):
        self.embedding_dims = embedding_dims
        self.indices = indices
        self.num_clusters = num_clusters
        self.num_threads = num_threads
        self.index = [KMeans(n_clusters=self.num_clusters,
                             n_jobs=self.num_threads,
                             n_init=n_init, max_iter=max_iter)
                      for _ in enumerate(self.indices)]

    def fit(self, embeddings, labels):
        """
            Cluster given data for given label indices in given number
            of clusters
            Args:
                embeddings: numpy.ndarray: document embeddings
                labels: csr_matrix: (num_samples, num_labels)
        """
        for idx, ind in enumerate(self.indices):
            doc_indices = labels[:, ind].nonzero()[0]
            if doc_indices.size < self.num_clusters:
                self.index[idx].cluster_centers_ = np.tile(
                    np.sum(embeddings[doc_indices, :], axis=0, keepdims=True),
                    (self.num_clusters, 1))
            else:
                self.index[idx].fit(embeddings[doc_indices, :])

    def predict(self):
        """
            Return cluster centroids
        """
        output = np.zeros((self.indices.size*self.num_clusters,
                           self.embedding_dims), dtype=np.float32)
        for idx, _ in enumerate(self.indices):
            ind = list(range(idx*self.num_clusters, (idx+1)*self.num_clusters))
            output[ind, :] = self.index[idx].cluster_centers_
        return output

    def save(self, fname):
        with open(fname, 'wb') as fp:
            pickle.dump({'embedding_dims': self.embedding_dims,
                         'index': self.index,
                         'num_clusters': self.num_clusters,
                         'num_sets': self.num_sets,
                         'num_threads': self.num_threads
                         }, fp
                        )

    def load(self, fname):
        with open(fname, 'rb') as fp:
            temp = pickle.load(fp)
            self.index = temp['index']
            self.embedding_dims = temp['embedding_dims']
            self.num_clusters = temp['num_clusters']
            self.num_threads = temp['num_threads']
            self.num_sets = temp['num_sets']
