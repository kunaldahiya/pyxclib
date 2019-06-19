import sys
from functools import partial
import numpy as np
import pickle as p
import scipy.sparse as sp
import os
from xclib.data import data_utils as du
from bs4 import BeautifulSoup
import re
dataset = p.load(open(sys.argv[1], 'rb'))
key = sys.argv[2]
data_dir = os.path.dirname(sys.argv[1])
lbs = dataset[key][1].tocsc()
lbs[lbs == -1] = 0
sort_freq = np.argsort(np.ravel(lbs.sum(axis=0)))
max_diff = 70000
diff = 2*max_diff
lbs = lbs.transpose()
num_instances = lbs.shape[1]
num_lbs = lbs.shape[0]
frac = 0.3
p_diff = 10*max_diff+2
iter_count = 0
lbs = lbs.tolil()
_lbs = lbs.tocsc()
while diff > max_diff:
    iter_count += 1
    test_indices = {}
    for indices in lbs[sort_freq, :].__dict__['rows']:
        flag = np.sum(list(map(lambda x: test_indices.get(x, False), indices)))
        if flag == 0:
            total = max(int(frac*len(indices)), 1)
            keys = np.random.choice(indices, size=total)
            test_indices.update(dict((key, True) for key in keys))

    test_idx = np.asarray(list(test_indices.keys()))
    train_idx = np.setdiff1d(np.arange(num_instances), test_idx)
    valid_lb_tst = np.where(np.ravel(_lbs[:, test_idx].sum(axis=1)) > 0)[0]
    valid_lb_trn = np.where(np.ravel(_lbs[:, train_idx].sum(axis=1)) > 0)[0]
    diff = num_lbs - np.intersect1d(valid_lb_tst,
                                    valid_lb_trn, assume_unique=True).shape[0]
    if diff < p_diff:
        split = np.zeros((num_instances), np.int32)
        split[test_idx] = 1
        np.savetxt(os.path.join(data_dir, 'split.0.txt'), split, fmt="%d")
        print("%d: diff: %d trn: %d tst: %d Labels: %d" %
              (iter_count, diff, train_idx.shape[0], test_idx.shape[0], num_lbs))
        p_diff = diff
    if iter_count > 100:
        break
    print("Cout: %d, diff: %d" % (iter_count, diff), end='\r')

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

split = np.loadtxt(os.path.join(data_dir, 'split.0.txt'), dtype=np.int32)
tst_instances = np.sum(split)
cmp_fts = open(os.path.join(data_dir, 'corpus.txt'), 'w')
tst_idx = 0
trn_idx = 0
fts = cleanhtml(('My_^_$'.join(dataset[key][0])).replace('\n','')).split('My_^_$')
print("Printing data")
lbs = dataset[key][1].tocsr()
fts_key = dataset[key][2]
print(len(fts), lbs.shape)
tst_idxs = np.where(split == 1)[0]
trn_idxs = np.where(split == 0)[0]

print(len(tst_idxs), len(trn_idxs))
tst_lbs = lbs[tst_idxs]
trn_lbs = lbs[trn_idxs]
for idx, flag in enumerate(split):
    print("%s->%s" % (fts_key[idx], fts[idx]), file=cmp_fts)
    if idx%1000 == 1 :
        print("[%d/%d]"%(idx, num_instances), end='\r')

print("[%d/%d]"%(idx, num_instances))
cmp_fts.close()
du.write_sparse_file(tst_lbs, os.path.join(
    data_dir, 'tst_lbl_mat.txt'))
du.write_sparse_file(trn_lbs, os.path.join(
    data_dir, 'trn_lbl_mat.txt'))
