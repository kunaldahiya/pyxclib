"""
    Compute evaluation statistics. 
"""

__author__='X'

import numpy as np

def format(*args, decimal_points='%0.2f'):
    out = []
    for vals in args:
        out.append(','.join(list(map(lambda x: decimal_points%(x*100), vals))))
    return '\n'.join(out)

class Metrices(object):
    def __init__(self, true_labels, remove_invalid=False):
        """
            Args:
                true_labels: csr_matrix: true labels with shape (num_instances, num_labels)
                remove_invalid: boolean: remove samples without any true label
        """
        self.true_labels = true_labels
        self.num_instances, self.num_labels = true_labels.shape
        self.remove_invalid = remove_invalid
        self.valid_idx = None
        if self.remove_invalid:
            samples = np.sum(self.true_labels,axis=1)
            self.valid_idx = np.arange(self.num_instances).reshape(-1,1)[samples>0]
            self.true_labels = self.true_labels[self.valid_idx]
            self.num_instances = self.valid_idx.size
        self.ndcg_denominator = np.cumsum(1/np.log2(np.arange(1,self.num_labels+1)+1)).reshape(-1,1)
        self.labels_documents = np.ravel(np.sum(self.true_labels, axis=1))
        self.labels_documents[self.labels_documents==0]=1

    def _rank_sparse(self, X, K):
        """
            Args:
                X: csr_matrix: sparse score matrix with shape (num_instances, num_labels)
                K: int: Top-k values to rank
            Returns: 
                predicted: np.ndarray: (num_instances, K) top-k ranks
        """
        total = X.shape[0]
        predicted = np.zeros((total, K), np.int32)
        for i, x in enumerate(X):
            index = x.__dict__['indices']
            data = x.__dict__['data']
            predicted[i] = index[np.argsort(-data)[0:K]]
        return predicted

    def eval(self, predicted_labels, K=5):
        """
            Args:
                predicted_labels: csr_matrix: predicted labels with shape (num_instances, num_labels)
                K: int: compute values from 1-5
        """
        if self.valid_idx is not None:
            predicted_labels = predicted_labels[self.valid_idx]
        assert predicted_labels.shape == self.true_labels.shape
        predicted_labels = self._rank_sparse(predicted_labels, K)
        prec = self.precision(predicted_labels, K)
        ndcg = self.nDCG(predicted_labels, K)
        return prec, ndcg

    def precision(self, predicted_labels, K):
        """
            Compute precision for 1-K
        """
        p= np.zeros((1,K))
        total_samples=self.true_labels.shape[0]
        ids = np.arange(total_samples).reshape(-1,1)
        p=np.sum(self.true_labels[ids, predicted_labels],axis=0)
        p = p/(total_samples)
        p = np.cumsum(p)/(np.arange(K)+1)
        return np.ravel(p)

    def nDCG(self, predicted_labels, K):
        """
            Compute nDCG for 1-K
        """
        ndcg= np.zeros((1, K))
        total_samples=self.true_labels.shape[0]
        ids = np.arange(total_samples).reshape(-1,1)
        dcg= self.true_labels[ids, predicted_labels]/(np.log2(np.arange(1,K+1)+1)).reshape(1,-1)
        dcg = np.cumsum(dcg,axis=1)
        denominator = self.ndcg_denominator[self.labels_documents-1]
        for k in range(K):
            temp = denominator.copy()
            temp[denominator>self.ndcg_denominator[k]] = self.ndcg_denominator[k]
            ndcg[0, k] = np.mean(dcg[:, k]/temp)
        return np.ravel(ndcg)

