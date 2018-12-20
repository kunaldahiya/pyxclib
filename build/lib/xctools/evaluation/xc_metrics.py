import scipy.sparse as sp
import numpy as np
"""
	Compute evaluation statistics.
"""

__author__ = 'X'


def format(*args, decimal_points='%0.2f'):
	out = []
	for vals in args:
		out.append(','.join(list(map(lambda x: decimal_points % (x*100), vals))))
	return '\n'.join(out)


def compute_inv_propesity(labels, A, B):
	num_instances, num_labels = labels.shape
	freqs = np.ravel(np.sum(labels, axis=0))
	C = (np.log(num_instances)-1)*np.power(B+1, A)
	wts = 1.0 + C*np.power(freqs+B, -A)
	return np.ravel(wts)


class Metrices(object):
	def __init__(self, true_labels, inv_propensity_scores=None, remove_invalid=False,batch_size=20):
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
			samples = np.sum(self.true_labels, axis=1)
			self.valid_idx = np.arange(self.num_instances).reshape(-1, 1)[samples > 0]
			self.true_labels = self.true_labels[self.valid_idx]
			self.num_instances = self.valid_idx.size
		"inserting dummpy index"
		self.true_labels_padded = sp.hstack(
		    [self.true_labels, sp.csr_matrix(np.zeros((self.num_instances, 1)))]).tocsr()
		self.ndcg_denominator = np.cumsum(
			1/np.log2(np.arange(1, self.num_labels+1)+1)).reshape(-1, 1)
		self.labels_documents = np.ravel(np.sum(self.true_labels, axis=1))
		self.labels_documents[self.labels_documents == 0] = 1
		self.inv_propensity_scores = None
		self.batch_size = batch_size
		if inv_propensity_scores is not None:
			self.inv_propensity_scores = np.hstack(
				[inv_propensity_scores, np.zeros((1))])
			assert(self.inv_propensity_scores.size == self.true_labels_padded.shape[1])

	def _rank_sparse(self, X, K):
		"""
			Args:
				X: csr_matrix: sparse score matrix with shape (num_instances, num_labels)
				K: int: Top-k values to rank
			Returns:
				predicted: np.ndarray: (num_instances, K) top-k ranks
		"""
		total = X.shape[0]
		labels = X.shape[1]

		predicted = np.full((total, K), labels)
		for i, x in enumerate(X):
			index = x.__dict__['indices']
			data = x.__dict__['data']
			idx = np.argsort(-data)[0:K]
			predicted[i, :idx.shape[0]] = index[idx]
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
		if self.inv_propensity_scores is not None:
			wt_true_mat = self._rank_sparse(self.true_labels.dot(sp.spdiags(
				self.inv_propensity_scores[:-1], diags=0, m=self.num_labels, n=self.num_labels)), K)
			PSprecision = self.PSprecision(predicted_labels, K) / self.PSprecision(wt_true_mat, K)
			PSnDCG=self.PSnDCG(predicted_labels, K) / self.PSnDCG(wt_true_mat, K)
			return [prec, ndcg, PSprecision, PSnDCG]
		else:
			return [prec,ndcg]

	def precision(self, predicted_labels, K):
		"""
			Compute precision for 1-K
		"""
		p=np.zeros((1, K))
		total_samples=self.true_labels.shape[0]
		ids=np.arange(total_samples).reshape(-1, 1)
		p=np.sum(self.true_labels[ids, predicted_labels], axis = 0)
		p=p*1.0/(total_samples)
		p=np.cumsum(p)/(np.arange(K)+1)
		return np.ravel(p)

	def nDCG(self, predicted_labels, K):
		"""
			Compute nDCG for 1-K
		"""
		ndcg=np.zeros((1, K))
		total_samples=self.true_labels.shape[0]
		ids=np.arange(total_samples).reshape(-1, 1)
		dcg=self.true_labels[ids, predicted_labels] /(
			np.log2(np.arange(1, K+1)+1)).reshape(1, -1)
		dcg=np.cumsum(dcg, axis = 1)
		denominator=self.ndcg_denominator[self.labels_documents-1]
		for k in range(K):
			temp=denominator.copy()
			temp[denominator > self.ndcg_denominator[k]]=self.ndcg_denominator[k]
			temp = np.power(temp,-1.0)
			for batch in np.array_split(np.arange(total_samples),self.batch_size):
				dcg[batch,k]=np.ravel(np.multiply(dcg[batch,k],temp[batch]))
			ndcg[0, k] = np.mean(dcg[:,k])
			del temp
		return np.ravel(ndcg)

	def PSnDCG(self, predicted_labels, K):
		"""
			Compute PSnDCG for 1-K
		"""
		psndcg=np.zeros((1, K))
		total_samples=self.true_labels.shape[0]
		ids=np.arange(total_samples).reshape(-1, 1)

		ps_dcg=self.true_labels_padded[ids, predicted_labels].toarray(
		)*self.inv_propensity_scores[predicted_labels]/np.log2(np.arange(1, K+1)+1).reshape(1, -1)
		
		ps_dcg=np.cumsum(ps_dcg, axis = 1)
		denominator=self.ndcg_denominator[self.labels_documents-1]
		
		for k in range(K):
			temp=denominator.copy()
			temp[denominator > self.ndcg_denominator[k]]=self.ndcg_denominator[k]
			temp = np.power(temp,-1.0)
			for batch in np.array_split(np.arange(total_samples), self.batch_size):
				ps_dcg[batch,k]=ps_dcg[batch,k]*temp[batch,0]
			psndcg[0, k] = np.mean(ps_dcg[:,k])
			del temp
		return np.ravel(psndcg)

	def PSprecision(self, predicted_labels, K):
			"""
				Compute PSprecision for 1-K
			"""
			psp=np.zeros((1, K))
			total_samples=self.true_labels.shape[0]
			ids=np.arange(total_samples).reshape(-1, 1)
			_p=self.true_labels_padded[ids, predicted_labels].toarray(
			)*self.inv_propensity_scores[predicted_labels]
			psp=np.sum(_p, axis = 0)
			psp=psp*1.0/(total_samples)
			psp=np.cumsum(psp)/(np.arange(K)+1)
			return np.ravel(psp)
