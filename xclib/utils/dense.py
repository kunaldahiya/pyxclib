import numpy as np
from .sparse import normalize as _normalize
from .sparse import binarize as _binarize


def compute_centroid(X, Y, reduction='sum', binarize=True, copy=True):
    """
    Compute label centroids from sparse features
    * output is dense

    Arguments:
    ---------
    X: np.ndarray
        embedding of each document
    Y: scipy.sparse.csr_matrix
        ground truth
    reduction: str, optional (default='sum')
        take sum or average
    copy: boolean, optional (default=True)
        create copy when binarizing

    Returns:
    --------
    centroids: np.ndarray
        Centroid for each label
    """
    if binarize:
        Y = _binarize(Y, copy=copy)
    centroids = Y.transpose().dot(X)
    if reduction == 'sum':
        pass
    elif reduction == 'mean':
        freq = np.ravel(np.sum(Y, axis=0)).reshape(-1, 1)
        freq[freq == 0] = 1.0  # Avoid division by zero
        centroids = centroids/freq
    else:
        raise NotImplementedError(
            "Reduction {} not yet implemented.".format(reduction))
    return centroids


def compute_dense_features(X, embeddings, reduction='sum', normalize=True,
                           binarize=False, copy=False):
    """
    Compute dense features as per given sparse features and word embeddings

    Arguments:
    ----------
    features: csr_matrix
        sparse features
    word_embeddings: np.ndarray
        dense embedding for each token in vocabulary
    reduction: str, optional (default=sum)
        sum or average
    normalize: boolean, optional (default=True)
        normalize features
    binarize: boolean, optional (default=False)
        binarize features
    copy: boolean, optional (default=True)
        create copies when binarizing or normalizing

    Returns:
    --------
    document_embeddings: np.ndarray
        dense embedding for each document
    """
    # convert the features to binary
    if binarize:
        X = _binarize(X, copy=copy)

    # l2 normalize
    if normalize:
        X = _normalize(X, copy=copy)

    document_embeddings = X @ embeddings
    if reduction == 'sum':
        pass
    elif reduction == 'mean':
        temp = np.array(features.sum(axis=1))
        temp[temp == 0] = 1.0  # Avoid division by zero
        document_embeddings = document_embeddings/temp
    else:
        raise NotImplementedError(
            "Reduction {} not yet implemented.".format(reduction))
    return document_embeddings
