#! python3
import numpy as np
import pickle
import sys
root = sys.argv[1]
labels = np.loadtxt(root+"/sa.txt", delimiter='->', dtype=str)
num_instances = labels.shape[0]

_overall_labels ={}
for i,line in enumerate(labels):
	lbls = np.unique(line[1].split('-^-'))
	for lbl in lbls:
		score = _overall_labels.get(lbl,0) + 1
		_overall_labels[lbl] = score
	print("[%d/%d]"%(i+1,num_instances),end='\r')

_overall_labels = dict(sorted(_overall_labels.items(), key=lambda x: -x[1]))
total_labels_corpus = len(list(_overall_labels.keys()))

with open(root+"/Yf.txt",'w') as f:
	overall_labels = {}
	valid_lbs = 0
	for (k,v) in _overall_labels.items():
		if v>1:
			valid_lbs+=1
			print("%s->%d"%(k,v),file=f)
			overall_labels[k] = v


print("%d out of %d are possible for training"%(valid_lbs,total_labels_corpus))
print("Building data for zero-shot")

overall_labels_zero_shot = {}
for i,instance in enumerate(labels):
	flag = False
	lbls = np.unique(line[1].split('-^-'))
	for lb in lbs:
		if overall_labels.get(lb,0) != 0:
			flag = True
			break
	if flag:
		for lb in lbs:
			if overall_labels.get(lb,0) == 0:
				overall_labels_zero_shot[lb] = 1
	print("[%d/%d]"%(i+1,num_instances),end='\r')

overall_labels_zero_shot = dict(sorted(overall_labels_zero_shot.items(), key=lambda x: -x[1]))

fzero = open(root+"/Yf-zeroshot.txt",'w')
valid_lbs = 0
for (k,v) in overall_labels_zero_shot.items():
	if v==1:
		valid_lbs+=1
		print("%s->%d"%(k,v),file=fzero)

print("%d out of %d are Total Labels"%(valid_lbs,total_labels_corpus))
