from xctools.data import data_utils as du
from scipy.io import loadmat
import numpy as np
import scipy.sparse as sp
import sys
import matplotlib.pyplot as plt
import itertools
"""
1: Ground Truth
2: Train Files
3: Precision @ K
4: Image name.extension
i=3 to len(argv)
i: Label on plot
i+1: color marker
i+2: path to predicted scores

i=i+3
"""

plt.style.use('dark_background')
font = {'family': 'serif', 'weight': 'normal', 'size': 16}

_, tr_labels, tr_num_samples, _, tr_num_labels = du.read_data(sys.argv[2])
true_labels = du.binarize_labels(tr_labels, tr_num_labels)
true_labels_padded = sp.hstack(
    [true_labels, sp.csr_matrix(np.zeros((tr_num_samples, 1)))]).tocsr()
tr_freq = np.ravel(np.sum(true_labels_padded,axis=0))

def plot_data(X,Y,K,plot,label,cm):
	plot.plot(np.log(np.asarray(X)), Y, cm, label=label)
	

def _rank_sparse( X, K):
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

def get_contribution(true_labels, freq ,predictions_file,K=5):
    te_num_samples,te_num_labels=true_labels.shape
    print(loadmat(predictions_file)['predicted_labels'].shape)
    
    predicted_labels = _rank_sparse(loadmat(predictions_file)['predicted_labels'],K)
    
    idx = np.arange(te_num_samples).reshape(-1,1)
    true_labels_padded = sp.hstack(
        [true_labels, sp.csr_matrix(np.zeros((te_num_samples, 1)))]).tocsr()
    
    contrib_labels = sp.lil_matrix((te_num_samples,te_num_labels+1),dtype=np.int32)
    flag_lbl = true_labels_padded[idx, predicted_labels]

    for i in range(te_num_samples):
        contrib_labels[i,predicted_labels[i]]=flag_lbl[i]

    contrib = np.ravel(np.sum(contrib_labels,axis=0)/(np.sum(contrib_labels)))
    # sorted_idx = np.argsort(-freq)
    # freq_bin = freq[sorted_idx]
    # t_contrib = contrib[sorted_idx]
    freq_bin = np.unique(freq)
    freq_bin = freq_bin[freq_bin!=0]
    t_contrib = []
    for bin in freq_bin:
        t_contrib.append(np.sum(contrib[freq == bin]))
        # t_contrib.append(np.sum(contrib[freq == bin])/freq[freq == bin].shape[0])
    return zip(*sorted(zip(np.ravel(freq_bin),np.ravel(t_contrib))))

K=int(sys.argv[3])
for i in range(5,len(sys.argv),3):
    _,te_labels, _, _, te_num_labels = du.read_data(sys.argv[1])
    true_labels = du.binarize_labels(te_labels, te_num_labels)
    freq,contrib = get_contribution(true_labels,tr_freq,sys.argv[i+2])
    plot_data(freq,contrib,K,plt,label=sys.argv[i],cm=sys.argv[i+1])

plt.title('Labels Contributing in P@%d'%(K))
plt.ylabel('P@%d'%(K))
plt.xlabel('Frequency')
plt.legend()
plt.savefig('%s'%(sys.argv[4]))
