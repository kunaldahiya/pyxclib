import nltk
import sys
from nltk.corpus import stopwords
import numpy as np
porter = nltk.PorterStemmer()

pattern = r"""(?:[A-Z]\.)+|\d+(?:\.\d+)?%?|\w+(?:[-']\w+)*|(?:[+/\-@&*])"""

vocab_files = sys.argv[1]
raw_data = sys.argv[2]
pre_data = sys.argv[3]

stop = {}
for w in stopwords.words('english'):
	stop[w] = 1

vocab_idx = {}
idx_vocab = {}
with open(vocab_files,'r',encoding='latin-1') as f:
	for i,v in enumerate(f):
		vocab_idx[v.strip()]=str(i)
		idx_vocab[str(i)]=v.strip()
total_fts = i+1
print(total_fts)

flag = True
document = []
pre_document = []
debug = open('log.txt','w')

with open(raw_data,'r',encoding='latin-1') as f,open(pre_data,'r',encoding='latin-1') as fp:
	stats = fp.readline()
	while True:
		sentence = f.readline()
		pre_sent = np.unique(list(map(lambda x:idx_vocab[x.split(':')[0]],fp.readline().split(' ')[1:])))
		_,sentence = sentence.split('->')
		d = nltk.regexp_tokenize(sentence,pattern)
		temp = []
		for w in d:
			w = w.lower()
			w = porter.stem(w)
			if vocab_idx.get(w,False) and not stop.get(w,False):
				temp.append(w)
		temp = np.unique(temp)
		extra = np.setdiff1d(temp,pre_sent)
		print(extra)
		break
		
	print(len(np.unique(temp)))
	print(len(pre_sent))
	# print(np.unique(temp))