import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from clusters import *
import pickle as pkl
from utils import *
import sys
import os

data_path = sys.argv[1]
data_set = sys.argv[2]
model_name = sys.argv[3]
upto = 1000

save_model = '../../models/ANALYSIS/%s/'%(data_set)
os.makedirs('../../models/ANALYSIS/%s/'%(data_set),exist_ok=True)

save_data = '../../results/ANALYSIS/%s/'%(data_set)
os.makedirs('../../results/ANALYSIS/%s/'%(data_set),exist_ok=True)


train_Matfile = pkl.load(open(data_path+'tr_doc_embeddings_labels.pkl',"rb"))
tr_doc_files =train_Matfile['embeddings']
tr_labels = train_Matfile['labels']

test_Matfile = pkl.load(open(data_path+'ts_doc_embeddings_labels.pkl',"rb"))
ts_doc_files = test_Matfile['embeddings']
ts_labels = test_Matfile['labels']


if model_name == 'nn':
	ANN = NearestNeighbor(2)
elif model_name == 'wnn':
	ANN = NearestNeighbor(2,weights='weighted')
elif model_name == 'hnsw':
	ANN = HNSW(M=100,efC=300,efS=1000,num_threads=12)
else:
	print('model not found')
	exit(0)

if not os.path.exists(save_model+model_name):
	ANN.fit(tr_doc_files,tr_labels)
	ANN.save(save_model+model_name)
else:
	print('Loading Model')
	ANN.load(save_model+model_name)

print(ANN)

def plot_data(fig_data,title,model):
	plt.style.use('dark_background')
	font = {'family': 'serif', 'weight': 'normal', 'size': 16}
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(title, fontdict=font)
	x = np.arange(upto)+1.0
	ax.plot(x,calc_recall(tr_doc_files,tr_labels,model,upto),'-r',label='Train')
	ax.plot(x,calc_recall(ts_doc_files,ts_labels,model,upto),'-g',label='Test')

	# ax.plot(x,calc_precision(tr_doc_files,tr_labels,ANN,upto),'.r',label='Train')
	# ax.plot(x,calc_precision(ts_doc_files,ts_labels,ANN,upto),'.g',label='Test')

	ax.set_ylabel('Recall@k')
	ax.set_xlabel('k')
	# ax.set_xlim([-1,upto+1])
	ax.set_ylim([0.0,1.0])

	fig.legend(loc='upper right')
	fig.savefig(fig_data)

plot_data(save_data+model_name+sys.argv[4]+'.png',data_set,ANN)