import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, find, spdiags
import math
import pdb

def rank_sparse(X):
    '''
        Rank of each element in decreasing order (per-row)
        Ranking will start from one (with zero at zero entries)
    '''
    def sortCoo(X):
        tuples = zip(X.row, X.col, X.data)
        return sorted(tuples, key=lambda x: (x[0], x[2]), reverse=True)
    temp = sortCoo(X.tocoo())
    rank_matrix = lil_matrix(X.shape, dtype=np.int)
    prev_row = X.shape[0]+1
    curr_rank = 0
    for item in temp:
        row_changed = prev_row!=item[0]
        if row_changed:
            prev_row = item[0]
            curr_rank = 1
        rank_matrix[item[0], item[1]] = curr_rank
        curr_rank +=1
    return rank_matrix.tocsr()


class Metrices(object):
    def __init__(self, labels, predicted_labels):
        self.labels = labels
        self.predicted_labels = predicted_labels
        assert self.predicted_labels.shape == self.labels.shape
        self.num_instances, self.num_labels = predicted_labels.shape

    def precision(self, K):
        """
            Compute precision for 1-K
        """
        rank_matrix = rank_sparse(self.predicted_labels)
        prec = []
        for k in range(1, K+1):
            temp = self.predicted_labels.copy()
            temp[rank_matrix>k] = 0
            temp = temp.multiply(self.labels)
            temp.eliminate_zeros()
            temp[temp!=0] = 1
            precision_at_k = np.array(temp.sum(axis=1))
            precision_at_k = precision_at_k/k
            prec.append(np.mean(precision_at_k))
        return prec

    def nDCG(self, K):
        """
            Compute nDCG for 1-K
        """
        rank_matrix = rank_sparse(self.predicted_labels)
        num_samples, num_labels = self.labels.shape
        ndcg = []
        wts = 1/np.log2(np.arange(1, num_labels+1)+1)
        cum_wts = np.cumsum(wts)
        
        rows, cols, vals = find(rank_matrix)
        temp = 1 / np.log2(vals + 1)
        coeff_mat = csr_matrix((temp, (rows, cols)), shape=(num_samples, num_labels), dtype=np.float32)
        for k in range(1, K+1):
            temp = coeff_mat.copy()
            temp[rank_matrix>k] = 0
            temp = temp.multiply(self.labels)
            temp.eliminate_zeros()            
            num = np.array(temp.sum(axis=1))
            count = np.array(self.labels.sum(axis=1))
            count = np.minimum(count, k)
            count[count==0] = 1
            den = cum_wts[count-1]
            ndcg.append(np.mean(num/den))
        return ndcg

    
    def compute_all(self, K=5):
        precision_at_k = self.precision(K)
        ndcg_at_k = self.nDCG(K)
        print(precision_at_k, ndcg_at_k)
