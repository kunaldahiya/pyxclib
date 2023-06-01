import random
from operator import itemgetter
from .ann import NearestNeighbor
from .sparse import frequency, retain_topk
import matplotlib.pyplot as plt
import numpy as np



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


def _split_based_on_frequency(freq, num_splits):
    """
        Split labels based on frequency
    """
    thresh = np.sum(freq)//num_splits
    index = np.argsort(-freq)
    indx = [[]]
    cluster_frq = 0
    t_splits = num_splits -1
    for idx in index:
        cluster_frq += freq[idx]
        if cluster_frq > thresh and t_splits > 0:
            t_splits-=1
            cluster_frq = freq[idx]
            indx[-1] = np.asarray(indx[-1])
            indx.append([])
        indx[-1].append(idx)
    indx[-1] = np.asarray(indx[-1])
    freq[freq == 0] = np.nan
    xticks = [f"{i+1}\n(#{freq[x].size//1000}K)\n({np.nanmean(freq[x]):.2f})" \
              for i, x in enumerate(indx)]
    return indx, xticks


def _pointwise_eval(predictions, true_labels, k, metric="P"):
    """
    Pointwise evaluation for different metrics, i.e., {P, R, and N}@k
    """    
    num_instances, num_labels = true_labels.shape
    freq = true_labels.sum(axis=0)
    scores = {}
    for key, val in predictions.items():
        v = retain_topk(val, k=k)
        v = v.multiply(true_labels)
        v.eliminate_zeros()
        v.data[:] = 1

        if metric == "P":
            v = v.multiply(1/(k*num_instances))
        elif metric == "R":
            deno = true_labels.sum(axis=1)*num_instances
            v = v.multiply(1/deno)
        elif metric == "%FN":
            v = true_labels - v
            v.eliminate_zeros()
            v = v.multiply(1/(freq*num_labels))
        else:
            raise NotImplementedError("Unknown metric!")
        
        scores[key] = np.ravel(v.sum(axis=0))
    return scores


def plot_group_hist(scores, xlabel, ylabel,
                    colors=None, opacity=[1.0, 0.5], dark=False, 
                    title='', fname='temp.eps'):
    """
    Plot group histograms 
    (e.g., decile wise plots, accuracy on different datasets)

    Arguments:
    ---------
    scores: {str: np.ndarray}
        score of each method at each point e.g., @1, 3, 5
    xlabel: [str]
        list containing label for each point on x-axis
    ylabel: str
        label on y-axis
    colors: list or None, optional (default=None)
        colors for each method (length must be same as len(scores))
    opacity: list of floats, optional (default=[1.0, 0.5])
        alpha in plot and grid
    dark: boolean, optional (default=False)
        use dark background
    title: str, optional (default='')
        title of the plot
    fname: str, optional (default='temp.eps')
        file name of the output file
    """
    if dark:
        plt.style.use('dark_background')
    len_methods = len(scores)
    n_groups = len(xlabel)
    _, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.95/(n_groups)
    shift = -bar_width*(len_methods-1)

    plt.grid(b=True, which='major', axis='both', alpha=opacity[1])
    for i, (_name, _val) in enumerate(scores.items()):
        _x = index+shift
        if colors is None:
            ax.bar(x=_x[::-1], height=_val, width=bar_width,
                alpha=opacity[0], label=_name)
        else:
            ax.bar(x=_x[::-1], height=_val, width=bar_width,
                alpha=opacity[0], label=_name, color=colors[i])

        shift += bar_width

    plt.xlabel('Quantiles \n (Increasing Freq.)', fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(title, fontsize=22)
    plt.xticks(index-bar_width*((len_methods-1)/2), xlabel[::-1], fontsize=14)
    plt.legend(fontsize=10)
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.0, dpi=1024)


def _decile_plot(scores, doc_freq, num_splits, ylabel="P",
                 title="Dataset", colors=None, opacity=[1.0, 0.5], dark=False,
                 fname="test.pdf"):
    indx, xlabel = _split_based_on_frequency(doc_freq, num_splits)
    contribs = {}
    for key, val in scores.items():
        contribs[key] = []
        for idx in indx:
            contribs[key].append(np.sum(val[idx])*100)
        contribs[key].append(np.sum(val)*100)
    xlabel += [f"complete\n(#{doc_freq.size//1000}K)"]
    plot_group_hist(
        scores=contribs,
        xlabel=xlabel,
        ylabel=f"{ylabel}",
        title=title,
        colors=colors,
        fname=fname,
        dark=dark,
        opacity=opacity)


def decile_contribution_plot(predictions, true_labels, train_labels,
                             k, num_splits, metric="P", title="Dataset",
                             colors=None, dark=False,
                             opacity=[1.0, 0.5], fname="test.pdf"):    
    """
    Plot decile wise contribution for different methods 
    (e.g., decile wise plots, accuracy on different datasets)

    Arguments:
    ---------
    predictions: {str: csr_matrix}
        predictions of each method
        * key would be used as method name on the legened (format it properly)
    true_labels: csr_matrix
        ground truth
    true_labels: csr_matrix
        train labels (used for computing frequencies)
    k: int
        compute contributions up to this point
    num_splits: int
        split labels in to these many bins
    metric: str, optional (default="P")
        metric for contribution
    title: str, optional (default='')
        title of the plot
    colors: list or None, optional (default=None)
        colors for each method (length must be same as len(scores))
    opacity: list of floats, optional (default=[1.0, 0.5])
        alpha in plot and grid
    dark: boolean, optional (default=False)
        use dark background
    fname: str, optional (default='temp.eps')
        file name of the output file
    """
    scores = _pointwise_eval(predictions, true_labels, k, metric)
    doc_frq = np.ravel(train_labels.sum(axis=0))
    label = f"{metric}@{k}" if metric in ["P", "R"] else metric
    _decile_plot(
        scores=scores,
        doc_freq=doc_frq,
        num_splits=num_splits,
        ylabel=label,
        title=title,
        colors=colors,
        dark=dark,
        opacity=opacity,
        fname=fname)
