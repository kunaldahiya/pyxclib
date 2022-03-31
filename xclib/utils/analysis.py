import random
from operator import itemgetter
from xclib.utils.ann import NearestNeighbor
from xclib.utils.sparse import frequency


def _sort_kv(ind, vals):
    temp = sorted(zip(ind, vals), key=lambda x: x[1], reverse=True)
    ind, vals = map(list, zip(*temp))
    return ind, vals


def _as_string(ind, vals, text, gt_ind, freq):
    """Represent key, val pairs as a string
    """
    def get_status(a, b):
        return "C" if a in b else "W"
    
    output = []
    gt_ind = set(gt_ind)
    
    if freq is None:
        for i, (k, v) in enumerate(zip(ind, vals)):
            output.append(f"{text[i]}: {v:.2f} ({get_status(k, gt_ind)})")
    else:
        for i, (k, v, f) in enumerate(zip(ind, vals, freq)):
            output.append(f"{text[i]}: {v:.2f} ({get_status(k, gt_ind)}, {f})")
    return ", ".join(output)


def get_random_indices(size, num_samples=1):
    return [random.randint(0, size-1) for _ in range(num_samples)]


def compare_predictions(doc_text, label_text, true_labels, predicted_labels,
                        train_labels=None, sample_indices=None, n_samples=10):
    """Print predictions for qualitative analysis
    Parameters
    ---------
    doc_text: list of str
        text of documents
    label_text: list of str
        text of labels
    true_labels: csr_matrix
        true labels with shape (num_samples, num_labels)
    predicted_labels: dict of csr_matrix
        multiple predicted labels with shape (num_samples, num_labels)
        method is identified using keys
    train_labels: csr_matrix, optional, default=None
        train labels (used to compute frequency)
    sample_indices: iterator, optional, default=None
        Analyze for these samples
    n_samples: int, optional, default=10
        Analyze for these many random samples
        * used only when sample_indices is none 
    """
    def process_one(_pred, _true, ind, text, freq):
        i, s = _pred[ind].indices, _pred[ind].data
        i, s = _sort_kv(i, s)
        f = None if freq is None else freq[i]
        return _as_string(
            i, s, itemgetter(*i)(text), _true[ind].indices, f)

    freq = None
    if train_labels is not None:
        # get #train documents for each label
        freq = frequency(train_labels, axis=0, copy=True).astype('int')

    if sample_indices is None:
        sample_indices = get_random_indices(len(doc_text), n_samples)

    for _, i in enumerate(sample_indices):
        _true = itemgetter(*true_labels[i].indices)(label_text)
        _true = ", ".join((_true, ) if isinstance(_true, str) else _true)
        _pred = ""
        for k, v in predicted_labels.items():
            _pred_one = process_one(v, true_labels, i, label_text, freq)
            _pred += f"{k}: {_pred_one}\n\n"
        print(f"text: {doc_text[i]}\n\ntrue labels: {_true}\n\n{_pred}----\n")


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
