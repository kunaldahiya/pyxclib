from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np
import sys

cachedStopWords = []
# cachedStopWords+=stopwords.words("english") 
# cachedStopWords+= '\' \" : ; , < > . ~ ` ! @ # ^ & * ( ) + = - _ { } [ ] '.split(' ')

vocab_files = sys.argv[1]
raw_data = sys.argv[2]
pre_data = sys.argv[3]

vocab_idx = {}
idx_vocab = {}

with open(vocab_files,'r',encoding='latin-1') as f:
	for i,v in enumerate(f):
		vocab_idx[v.strip()]=str(i)
		idx_vocab['%d'%(i)]=v.strip()

total_fts = i+1

flag = True
document = []
pre_document = []
debug = open('log.txt','w')
with open(raw_data,'r',encoding='latin-1') as fr, open(pre_data,'r',encoding='latin-1') as fp:
	stats = list(map(int,fp.readline().split(' ')))
	k=0
	while k<stats[0]:
		sr = fr.readline()
		sp = fp.readline()
		document.append([])
		for words in sr.lower().strip().split(' '):
			id = vocab_idx.get(words,'')
			print("%s,%s"%(words,id),file=debug)
			if id !='':
				document[-1].append(id)

		document[-1]=np.unique(document[-1])
		pre_document.append([])

		for words_idf in sp.lower().strip().split(' ')[1:]:
			words,idf = words_idf.split(':')
			pre_document[-1].append(words)

		if len(document[-1]) < len(pre_document[-1]):
			print([ id for id in document[-1]])
			print([ idx_vocab[id] for id in document[-1]])
			print([ idx_vocab.get(id) for id in pre_document[-1]])
			print(len(document[-1]))
			print(len(pre_document[-1]))
			print(k)
			print(sr)
			break
		break
		print("[%06d]%3d,%3d"%(k,len(document[-1]),len(pre_document[-1])),end='\r')
		k+=1
print([ idx_vocab[id] for id in document[-1]])
print([ idx_vocab.get(id) for id in pre_document[-1]])
print(len(document[-1]))
print(len(pre_document[-1]))

print()
print(total_fts)