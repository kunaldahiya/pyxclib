#! python3

import numpy as np
from scipy.sparse import lil_matrix
from xctools.data import data_utils as du
import sys
root = sys.argv[1]

labels = np.loadtxt(root+"/sa.txt", delimiter='->', dtype=str)
num_instances = labels.shape[0]

dictionary = np.loadtxt(root+"/Yf.txt", delimiter='->', dtype=str)
num_lbs = dictionary.shape[0]
label_idx = {}
for i,(k,v) in enumerate(dictionary):
	label_idx[k] = i

labels_sparse_mat = lil_matrix((num_instances,num_lbs+1), dtype=np.int32)
for i,instance in enumerate(labels):
	lbs = instance[1].split('-^-')
	for lb in lbs:
		lb_idx = label_idx.get(lb,num_lbs)
		labels_sparse_mat[i,lb_idx] = 1
	print("[%d/%d]"%(i+1,num_instances),end='\r')

labels_sparse_mat = labels_sparse_mat[:,:-1].tocsr()
print()
valid_instances = np.where(np.ravel(labels_sparse_mat.sum(axis=1))!=0)[0]
total_valid_instances = valid_instances.size
np.savetxt("valid_idx.txt",valid_instances,fmt="%d")
print("Total Documents are %d"%(total_valid_instances))
labels_sparse_mat = labels_sparse_mat[valid_instances]

print("min data",np.min(labels_sparse_mat.sum(axis=0)))

du.write_sparse_file(filename="corpus_lbl_mat.txt",labels=labels_sparse_mat)
k = 0
with open(root+"/text.txt",'r') as f, open(root+"/title.txt",'r') as ft, open(root+"/corpus_text.txt","w") as out, open(root+"/corpus_titles.txt","w") as outt:
	for i, line in enumerate(f):
		titles = ft.readline()
		if k<total_valid_instances and i == valid_instances[k]:
			print(line.strip(),file=out)
			print(titles.strip(),file=outt)
			k+=1
		print("[%d/%d]"%(i+1,num_instances), end="\r")


print("Building for zero-shot")
dictionary = np.loadtxt(root+"/Yf-zeroshot.txt", delimiter='->', dtype=str)
num_instances = labels.shape[0]
num_lbs = dictionary.shape[0]

label_idx = {}
for i,(k,v) in enumerate(dictionary):
	label_idx[k] = i

labels_sparse_mat = lil_matrix((num_instances,num_lbs+1), dtype=np.int32)

for i,instance in enumerate(labels):
	lbs = instance[1].split('-^-')
	for lb in lbs:
		lb_idx = label_idx.get(lb,num_lbs)
		labels_sparse_mat[i,lb_idx] = 1
	print("[%d/%d]"%(i+1,num_instances),end='\r')

labels_sparse_mat = labels_sparse_mat[:,:-1].tocsr()
labels_sparse_mat = labels_sparse_mat[valid_instances]

du.write_sparse_file(filename=root+"/corpus_lbl_zeroshot_mat.txt",labels=labels_sparse_mat)
