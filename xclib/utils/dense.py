import numpy as np
from .sparse import normalize as _normalize
from .sparse import binarize as _binarize


def compute_dense_features(features, word_embeddings, method='wt_sum',
                           normalize=True, binarize=False):
    """
    Compute dense features as per given sparse features and word embeddings

    Arguments:
    ----------
    features: csr_matrix
        sparse features
    word_embeddings: np.ndarray
        dense embedding for each token in vocabulary
    method: str, optional (default=wt_sum)
        wt_sum or wt_avg
    normalize: boolean, optional (default=True)
        normalize features
    binarize: boolean, optional (default=False)
        binarize features

    Returns:
    --------
    document_embeddings: np.ndarray
        dense embedding for each document
    """
    # convert the features to binary
    if binarize:
        features = _binarize(features, copy=True)

    # l2 normalize
    if normalize:
        features = _normalize(features, copy=True)

    document_embeddings = features @ word_embeddings
    if method == 'wt_avg':
        temp = np.array(features.sum(axis=1))+1e-5
        document_embeddings = document_embeddings/temp
    return document_embeddings
