import sys
import _pickle as cPickle

root = sys.argv[1]
text_file_trn = root+'trn_ft.txt'
labl_file_trn = root+'trn_lbl_mat.txt'
text_file_tst = root+'tst_ft.txt'
labl_file_tst = root+'tst_lbl_mat.txt'
vocab_file = root+'words.txt'

trn_txt = open(text_file_trn,'r')
trn_lbl = open(labl_file_trn,'r')

instances,labels = map(int,trn_lbl.readline().split(' '))
print(labels)
trn_data = []
split = 'train'
for inst in range(instances):
	inst_dict = {}
	inst_dict['text'] = trn_txt.readline().strip()
	inst_dict['split'] = split
	inst_dict['num_words'] = len(inst_dict['text'].split(' '))
	inst_dict['catgy'] = list(map(int,trn_lbl.readline().strip().split(',')))
	inst_dict['Id'] = str(inst)
	trn_data.append(inst_dict)
	print("[%d/%d]"%(inst,instances),end='\r')

tst_txt = open(text_file_tst,'r')
tst_lbl = open(labl_file_tst,'r')

instances,labels = map(int,tst_lbl.readline().split(' '))
tst_data = []
split = 'test'
for inst in range(instances):
	inst_dict = {}
	inst_dict['text'] = tst_txt.readline().strip()
	inst_dict['split'] = split
	inst_dict['num_words'] = len(inst_dict['text'].split(' '))
	inst_dict['catgy'] = list(map(int,tst_lbl.readline().strip().split(',')))
	inst_dict['Id'] = str(inst)
	tst_data.append(inst_dict)
	print("[%d/%d]"%(inst,instances),end='\r')

vocab = {}
with open(vocab_file,'r') as f:
	for i,word in enumerate(f.readlines()):
		vocab[word.strip()]=i*1.0

catgy = {}
for i in range(labels):
	catgy[str(i)] = i

dataset = open('../pyXMLC/data/xml_data/'+sys.argv[2]+'.p','wb')
cPickle.dump([trn_data,tst_data,vocab,catgy], dataset,protocol=2)
dataset.close()