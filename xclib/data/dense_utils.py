import numpy as np

def compute_dense_features(features, word_embeddings, method='wt_sum'):
    """
        Compute dense features as per given sparse features and word embeddings
        Args:
            features: csr_matrix: sparse features
            word_embeddings: np.ndarray: dense embedding for each word in vocabulary
            method: str: wt_sum or wt_avg
        Returns:
            document_embeddings: np.ndarray: dense embedding for each document
    """
    document_embeddings = features.dot(word_embeddings)
    if method == 'wt_avg':
        temp = np.array(features.sum(axis=1))+1e-5
        document_embeddings = document_embeddings/temp
    elif method == 'wt_norm_l2':
        temp = features.power(2).sum(axis=1).sqrt()+1e-5
        document_embeddings = document_embeddings/temp 
    return document_embeddings
