import sys
import pickle as p
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')


def split_based_on_frequency(labels, num_splits, threshold=[7]):
	"""
	Split labels based on frequency
	"""
	freq = get_frequency(labels)
	if len(threshold) == 0:
		indx = np.array_split(np.argsort(freq), num_splits)
	else:
		indx = []
		threshold = [0] + threshold + [np.max(freq)+1]
		for idx, thresh in enumerate(threshold[1:]):
			indx.append(np.where(np.logical_and(
				freq >= threshold[idx], freq < thresh))[0])
		pass
	xticks = ["%d\n(#%d)" % (i+1, freq[x].size) for i, x in enumerate(indx)]
	return indx, xticks


def plot(val, plot_labels=['tail', 'head', 'full'], fname='temp.eps', title=''):
	"""
	Plot
	"""
	n_groups = len(plot_labels)
	fig, ax = plt.subplots()
	index = np.arange(n_groups)
	bar_width = 0.8
	opacity = 0.4
	ax.bar(x=index, height=val, width=bar_width,
		   alpha=opacity, label="Frequency of labels")
	ax.set_xlabel('K')
	ax.set_ylabel('#(Labels freq >=K )')
	ax.set_title(title)
	ax.set_xticks(index)
	ax.set_xticklabels(plot_labels)
	rects = ax.patches
	for rect, label in zip(rects, val):
		height = rect.get_height()
		ax.text(rect.get_x() + rect.get_width() / 2, height + 5, "%dK" %
				int(label/1000), ha='center', va='bottom')
	ax.legend()
	fig.tight_layout()
	plt.savefig(fname)


def _get_label_instance_distribution(lbs):
	K = [1, 2, 4, 8, 16, 32, 64]
	lbs[lbs == -1] = 1
	lbs = lbs.tocsc()
	freq_lbs = np.ravel(lbs.sum(axis=0))
	lbls = []
	valu = []
	fval = []
	for idx in K:
		lbls.append("%d" % (idx))
		valid_lbs = np.where(freq_lbs >= idx)[0]
		valu.append(valid_lbs.shape[0])
		_temp_mat = lbs[:, valid_lbs]
		_temp_mat = _temp_mat.tocsr()
		_temp_mat.eliminate_zeros()
		valid_ins = np.where(np.ravel(_temp_mat.sum(axis=1)) > 0)[0]
		fval.append(valid_ins.shape[0])
		print("\t",valu[-1], fval[-1])
	plot(valu, lbls, fname="labels_left", title="Hike user statistics")
	plot(fval, lbls, fname="documents_left", title="Hike user statistics")
	del lbs


def _get_stats(lbl_mat):
	_lbl_mat = lbl_mat.tocsr()
	_lbl_mat.__dict__['data'][:] = 1
	avg_lbl_per_insta = np.sum(_lbl_mat.sum(axis=1))/(_lbl_mat.shape[0]+1e-3)
	avg_insta_per_lbl = np.sum(_lbl_mat.sum(axis=0))/(_lbl_mat.shape[1]+1e-3)
	print("\t#Instances: %d\n\t#Labels: %d" % _lbl_mat.shape)
	print("\tavg lbl/insta: %f\n\tavg insta/lbl: %f," %
		  (avg_lbl_per_insta, avg_insta_per_lbl))
	del _lbl_mat


def _create_from_raw(dataset, features):
	print("Building Dataset")
	df = pd.read_csv(dataset)
	dff = pd.read_csv(features)
	features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f3']
	
	for feat in features:
		caterogries = dff[feat].unique()
		for cat in caterogries:
			dff['%s-%d' % (feat, cat)] = 0
			dff['%s-%d' % (feat, cat)][dff[feat] == cat] = 1
		dff.drop('%s'%(feat), axis='columns')

	groups = df.sort_values(by='node1_id').groupby('node1_id')

	labels = df['node2_id'].unique()
	index_id = dict((lb, i) for i, lb in enumerate(labels))
	ft_user = dict(map(lambda x: (x[0], x[1:]), dff.values))
	print(dff.shape)
	num_labels = labels.shape[0]
	num_instances = len(groups)

	lbs = lil_matrix((num_instances, num_labels), dtype=np.int32)

	data_stats = {
		"total": 0,
		"active": 0,
		"labels": 0
	}
	fts = []
	ftskeys = []
	for idx, (name, group) in enumerate(groups):
		data_stats["total"] += 1
		data_stats["active"] += group['is_chat'].sum()
		data_stats["labels"] += group.shape[0]
		l_idx, l_val = zip(*map(lambda x: (index_id[x[0]], x[1]), zip(
			group['node2_id'].values, group['is_chat'].values)))
		fts.append(ft_user[name])
		lbs[idx, l_idx] = l_val
		ftskeys.append(name)
		print("[%d/%d]"%(idx, num_instances), end='\r')
	lbfts = list(map(lambda x: ft_user[x], labels))
	print(" "*50, end='\r')
	print("\tTotal users:%d\n\tAvg active users: %0.3f\n\tAvg contacts: %0.3f" % (
		data_stats['total'], data_stats['active']/data_stats['total'], data_stats['labels']/data_stats['total']))
	return lbs, fts, lbfts, ftskeys


def _create_y_mat(lbs, fts, lbsfts, ftskeys):
	_lbl_mat = lbs.tocsc()
	_lbl_mat.__dict__['data'][:] = 1
	freq_lbs = np.ravel(_lbl_mat.sum(axis=0))
	valid_lbs = np.where(freq_lbs >= 2)[0]
	_temp_mat = _lbl_mat.tocsc()[:, valid_lbs]
	_temp_mat = _temp_mat.tocsr()
	_temp_mat.eliminate_zeros()
	valid_ins = np.where(np.ravel(_temp_mat.sum(axis=1)) > 0)[0]
	_lbs = lbs.tocsc()[:, valid_lbs]
	_lbs = _lbs.tocsr()[valid_ins, :]
	_fts = list(map(lambda x: fts[x], valid_ins))
	_lbfts = list(map(lambda x: lbsfts[x], valid_lbs))
	_ftskeys = list(map(lambda x: ftskeys[x], valid_ins))
	return {'Normal': (_fts, _lbs, _ftskeys, _lbfts)}


def _create_zero_y_mat(lbs, fts, lbsfts, ftskeys):
	_lbl_mat = lbs.tocsc()
	_lbl_mat.__dict__['data'][:] = 1
	freq_lbs = np.ravel(_lbl_mat.sum(axis=0))
	valid_lbs = np.where(freq_lbs == 1)[0]
	_temp_mat = _lbl_mat.tocsc()[:, valid_lbs]
	_temp_mat = _temp_mat.tocsr()
	_temp_mat.eliminate_zeros()
	valid_ins = np.where(np.ravel(_temp_mat.sum(axis=1)) > 0)[0]
	_lbs = lbs.tocsc()[:, valid_lbs]
	_lbs = _lbs.tocsr()[valid_ins, :]
	_fts = list(map(lambda x: fts[x], valid_ins))
	_lbfts = list(map(lambda x: lbsfts[x], valid_lbs))
	_ftskeys = list(map(lambda x: ftskeys[x], valid_ins))
	return {'Zero': (_fts, _lbs, _ftskeys, _lbfts)}


if __name__ == '__main__':
	dataset = sys.argv[1]
	features = sys.argv[2]
	if sys.argv[3] == "1":
		plt.style.use('dark_background')
		font = {'weight': 'normal', 'size': 14}
	else:
		font = {'weight': 'normal', 'size': 14}
	lbs, fts, lbfts, ftskeys = _create_from_raw(dataset, features)
	print("Getting Raw datastats")
	_get_stats(lbs)
	print("Plotting labels")
	_get_label_instance_distribution(lbs.copy())
	print("Building Y dataset")
	Normal = _create_y_mat(lbs, fts, lbfts, ftskeys)
	_get_stats(Normal["Normal"][1])
	with open('Y_Hike.pkl', 'wb') as f:
		p.dump(Normal, f)
	
	print("Building 0-Y dataset")
	Zerosh = _create_zero_y_mat(lbs, fts, lbfts, ftskeys)
	_get_stats(Zerosh["Zero"][1])
	with open('0_Y_Hike.pkl', 'wb') as f:
		p.dump(Zerosh, f)
