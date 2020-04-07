import numpy as np
import random
from operator import itemgetter
from xclib.utils.ann import NearestNeighbor


def _sort_kv(ind, vals):
    temp = sorted(zip(ind, vals), key=lambda x: x[1], reverse=True)
    ind, vals = map(list, zip(*temp))
    return ind, vals


def _as_string(ind, vals):
    """Represent key, val pairs as a string
    """
    if vals is None:
        return ", ".join(["{}".format(item) for item in ind])
    else:
        return ", ".join(["{}: {:.2f}".format(
            item[0], item[1]) for item in zip(ind, vals)])


def get_random_indices(size, num_samples=1):
    return [random.randint(0, size-1) for _ in range(num_samples)]


def compare_predictions(docs, labels, true_labels, predicted_labels,
                        sample_indices=None, num_samples=10):
    """Print predictions for qualitative analysis
    Parameters
    ---------
    docs: list of str
        text of documents
    labels: list of str
        text of labels
    true_labels: csr_matrix
        true labels with shape (num_samples, num_labels)
    predicted_labels: dict of csr_matrix
        multiple predicted labels with shape (num_samples, num_labels)
        method is identified using keys
    num_samples: int, optional, default=10
        Analyze for these many samples
    sample_indices: iterator, optional, default=None
        Analyze for these samples
    """
    if sample_indices is None:
        sample_indices = get_random_indices(len(docs), num_samples)

    for _, idx in enumerate(sample_indices):
        _true = itemgetter(*true_labels[idx].indices)(labels)
        _pred = ""
        for key, val in predicted_labels.items():
            _ind, _score = val[idx].indices, val[idx].data
            _ind, _score = _sort_kv(_ind, _score)
            _pred += "{}: {}\n\n".format(
                key, _as_string(itemgetter(*_ind)(labels), _score))

        print("text: {}\ntrue labels: {}\n\n{}-----\n".format(
            docs[idx], _true, _pred))


def compare_nearest_neighbors(tr_embedding, tr_text, ts_embedding=None,
                              ts_text=None, num_neighbours=10,
                              sample_indices=None, num_samples=10,
                              method='brute', space='cosine',
                              num_threads=-1):
    """Analyze nearest neighbors for given documents/words
    Parameters
    ---------
    tr_embedding: np.ndarray
        representation for training set
    tr_text: list of str
        raw text for training set
    ts_embedding: np.ndarray, optional, default=None
        representation for training set
    ts_text: list of str, optional, default=None
        raw text for text set
    num_neighbours: int, optional, default=10
        Get these many neighbors
    num_samples: int, optional, default=10
        Analyze for these many samples
    sample_indices: iterator, optional, default=None
        Analyze for these samples
    method: str, optional, default='brute'
        Method to use in Nearest neighbors
    space: str, optional, default='cosine'
        Compute neighbors in this space
    num_threads: int, optional, default=-1
        Number of threads to use
    """
    if ts_embedding is None:
        ts_embedding = tr_embedding
        ts_text = tr_text

    if sample_indices is None:
        sample_indices = get_random_indices(len(ts_text), num_samples)

    graph = NearestNeighbor(num_neighbours=num_neighbours,
                            method=method,
                            space=space,
                            num_threads=num_threads)
    graph.fit(tr_embedding)

    for _, idx in enumerate(sample_indices):
        ind, dist = graph.predict(ts_embedding[idx].reshape(1, -1))
        # Returns as list of list
        ind, dist = ind[0], dist[0]
        temp = _as_string(itemgetter(*ind)(tr_text), dist)
        print("Index: {}, Original text: {}, Neighbors: {}\n".format(
                idx, ts_text[idx], temp))
