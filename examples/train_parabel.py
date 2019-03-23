"""
Usage
python train_parabel.py train.txt test.txt model_directory num_tree mode[train|predict]
"""

__author__='X'

from xclib.classifier.parabel import Parabel as parabel
from xclib.data import data_utils as du
import numpy as np
import sys

if __name__=='__main__':
	trn_file = sys.argv[1]
	tst_file = sys.argv[2]
	model_dir = sys.argv[3]
	num_tree = int(sys.argv[4])
	output_score_mat = sys.argv[5]
	mode = sys.argv[6]
	P = parabel(model_dir, num_tree = num_tree)

	if mode =="train":
	    tr_fts, tr_lbs, tr_num_samples, _, tr_num_labels = du.read_data(trn_file)
	    trn_labels = du.binarize_labels(tr_lbs, tr_num_labels)
	    P.fit(tr_fts, trn_labels)

	if mode =="predict":
	    tst_fts, _, te_num_samples, _, te_num_labels = du.read_data(tst_file)    
	    score_mat = P.predict(tst_fts)
	    du.write_sparse_file(score_mat, output_score_mat)
	
