import sys
import numpy as np
from xclib.data import data_utils as du

root = sys.argv[1]
splits = np.loadtxt(root+'/split.0.txt',dtype=int)
Ylbl = du.read_sparse_file(root+'/corpus_lbl_mat.txt')

du.write_sparse_file(Ylbl[splits==1],root+'/tst_lbl_mat.txt')
du.write_sparse_file(Ylbl[splits==0],root+'/trn_lbl_mat.txt')

text = open(root+"/corpus_text.txt","r")
tst_text = open(root+"/test_map.txt","w")
trn_text = open(root+"/train_map.txt","w")


titles = open(root+"/corpus_titles.txt","r")
tst_titles = open(root+"/test_titles.txt","w")
trn_titles = open(root+"/train_titles.txt","w")

for idx,i in enumerate(splits):
	txt = text.readline().strip()
	ttl = titles.readline().strip()
	if i ==0:
		print(txt,file=trn_text)
		print(ttl,file=trn_titles)
	if i ==1:
		print(txt,file=tst_text)
		print(ttl,file=tst_titles)
	print("[%d/%d]"%(idx,splits.shape[0]),end='\r')
