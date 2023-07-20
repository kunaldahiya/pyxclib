# Use CPUs for computations

from sklearn.cluster import KMeans
import pickle
import numpy as np
from .sparse import normalize
import math
from joblib import Parallel, delayed
import itertools


class Cluster(object):
    """Cluster given data for given label indices in given number
    of clusters

    Arguments
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
                 max_iter=50, n_init=1, num_threads=-1):
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


def b_kmeans_dense(X, index, metric='cosine', tol=1e-4):
    X = normalize(X)
    n = X.shape[0]
    if X.shape[0] == 1:
        return [index]
    cluster = np.random.randint(low=0, high=X.shape[0], size=(2))
    while cluster[0] == cluster[1]:
        cluster = np.random.randint(
            low=0, high=X.shape[0], size=(2))
    _centeroids = X[cluster]
    _similarity = np.dot(X, _centeroids.T)
    old_sim, new_sim = -1000000, -2
    while new_sim - old_sim >= tol:
        clustered_lbs = np.array_split(
            np.argsort(_similarity[:, 1]-_similarity[:, 0]), 2)
        _centeroids = normalize(np.vstack([
            np.mean(X[x, :], axis=0) for x in clustered_lbs
        ]))
        _similarity = np.dot(X, _centeroids.T)
        old_sim, new_sim = new_sim, np.sum(
            [np.sum(
                _similarity[indx, i]
            ) for i, indx in enumerate(clustered_lbs)])/n

    return list(map(lambda x: index[x], clustered_lbs))


def b_kmeans_sparse(X, index, metric='cosine', tol=1e-4, leakage=None):
    if X.shape[0] == 1:
        return [index]
    cluster = np.random.randint(low=0, high=X.shape[0], size=(2))
    while cluster[0] == cluster[1]:
        cluster = np.random.randint(
            low=0, high=X.shape[0], size=(2))
    _centeroids = X[cluster].todense()
    _similarity = _sdist(X.A, _centeroids.A,
                         metric=metric, norm='l2')
    old_sim, new_sim = -1000000, -2
    while new_sim - old_sim >= tol:
        clustered_lbs = np.array_split(
            np.argsort(_similarity[:, 1]-_similarity[:, 0]), 2)
        _centeroids = np.vstack([
            X[x, :].mean(
                axis=0) for x in clustered_lbs
        ])
        _similarity = _sdist(X.A, _centeroids.A,
                             metric=metric, norm='l2')
        old_sim, new_sim = new_sim, np.sum(
            [np.sum(
                _similarity[indx, i]
            ) for i, indx in enumerate(clustered_lbs)])

    if leakage is not None:
        _distance = 1-_similarity
        # Upper boundary under which labels will co-exists
        ex_r = [(1+leakage)*np.max(_distance[indx, i])
                for i, indx in enumerate(clustered_lbs)]
        """
        Check for labels in 2nd cluster who are ex_r_0 closer to 
        1st Cluster and append them in first cluster
        """
        clustered_lbs = list(
            map(lambda x: np.concatenate(
                [clustered_lbs[x[0]],
                 x[1][_distance[x[1], x[0]] <= ex_r[x[0]]]
                 ]),
                enumerate(clustered_lbs[::-1])
                )
        )
    return list(map(lambda x: index[x], clustered_lbs))


def _sdist(XA, XB, metric, norm=None):
    if norm is not None:
        XA = normalize(XA, norm)
        XB = normalize(XB, norm)
    if metric == 'cosine':
        score = XA.dot(XB.transpose())
    return score


def cluster_balance(X, clusters, num_clusters, splitter,
                    num_threads=5, verbose=True, use_sth_till=-1):
    """
    Cluster given data using 2-Means++ algorithm
    Arguments:
    ----------
    X: np.ndarray or csr_matrix
        input data
    clusters: list
        input already clustered? - pass that
        consider each point as a cluster otherwise
    num_clusters: int
        cluster data into these many clusters
        * it'll convert to nearest power of 2, if not already
    splitter: a function to split data
        * dense data: b_kmeans_dense
        * sparse data: b_kmeans_sparse
    num_threads: int, optional (default=5)
        number of threads to use
        * it'll use a single thread to initial partitions to avoid memory error
    verbose: boolean, optional (default=True)
        print time taken etc
    use_sth_till: int, optional (default=-1)
        use single thread till these many clusters

    Returns:
    --------
    clusters: a list of list
        the sub-list contains the indices for the vectors which were clustered
        together
    mapping: np.ndarray
        a 1D array containing the cluster id of each item in X
    """
    def _nearest_two_power(x):
        return 2**int(math.ceil(math.log(x) / math.log(2)))

    def _print_stats(x):
        print(f"Total clusters {len(x)}" \
            f" Avg. Cluster size {np.mean(list(map(len, x)))}")

    num_clusters = _nearest_two_power(num_clusters)

    while len(clusters) < use_sth_till:
        temp_cluster_list = [splitter(X[x], x) for x in clusters]
        clusters = list(itertools.chain(*temp_cluster_list))
        if verbose:
            _print_stats(clusters)
        del temp_cluster_list

    with Parallel(n_jobs=num_threads, prefer="threads") as parallel:
        while len(clusters) < num_clusters:
            temp_cluster_list = parallel(
                delayed(splitter)(X[x], x) for x in clusters)
            clusters = list(itertools.chain(*temp_cluster_list))
            if verbose:
                _print_stats(clusters)
            del temp_cluster_list
    mapping = np.zeros(X.shape[0], dtype=np.int64)
    for idx, item in enumerate(clusters):
        for _item in item:
            mapping[_item] = idx
    return clusters, mapping
