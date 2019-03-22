#! python3
import sys
import numpy as np
from xctools.data import data_utils as du
root = sys.argv[1]

y = du.read_sparse_file(root+"/corpus_lbl_mat.txt")
maxdiff = 20000
num_lbls = y.shape[1]
num_instance = y.shape[0]
instances_id = np.arange(num_instance)
diff =  maxdiff+1
counter = 1

lb_index = np.asarray(np.argsort(np.ravel(y.sum(axis=0))),dtype=np.int32)
lb_doc = y.transpose()
docs = []

document_lbs = np.ravel(y.sum(axis=1))

for lb in lb_doc:
	docs.append(lb.__dict__['indices'])

diff_flag = 20000000
counter = 1
while counter <= 100:
	test_instances = {}
	flag = True
	for i,lb_idx in enumerate(lb_index):
		flag = True
		documents = docs[lb_idx]
		for document in documents:
			if test_instances.get(document,False):
				flag = False
				break
		if flag:
			split_num = 1
			p = document_lbs[documents]**-1
			p = p/np.sum(p)
			for ini in np.random.choice(documents, p=p,size=split_num).tolist():
				test_instances[ini] = True
		print("counter: %d [%d/%d]"%(counter, i+1,num_lbls),end='\r')

	test_instances = np.asarray(list(test_instances.keys()))
	train_instances = np.setdiff1d(np.arange(num_instance), test_instances)
	tst_lbs = np.where(np.ravel(y[test_instances].sum(axis=0))>0)[0]
	trn_lbs = np.where(np.ravel(y[train_instances].sum(axis=0))>0)[0]
	diff = np.intersect1d(tst_lbs,trn_lbs).size-num_lbls
	_flag = np.abs(trn_lbs.size-tst_lbs.size)
	if _flag > diff_flag:
		print()
		print("Train #: %d"%(train_instances.size))
		print("Test #: %d"%(test_instances.size))
		print(tst_lbs.size, trn_lbs.size)
		splits = np.ones(num_instance,dtype=np.int32)
		splits[train_instances] = 0
		np.savetxt(root+"/split.0.txt",splits,fmt="%d")
		diff_flag = _flag
		if _flag ==0:
			break
	counter+=1