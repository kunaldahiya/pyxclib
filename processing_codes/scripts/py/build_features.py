from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np
import sys
# 
cachedStopWords = []
# cachedStopWords+=stopwords.words("english")
# cachedStopWords+= '\' \" : ; , < > . ~ ` ! @ # ^ & * ( ) + = - _ { } [ ] '.split(' ')

root_a = sys.argv[1]
root_b = sys.argv[2]
root_c = sys.argv[3]

vocab_files = sys.argv[4]
flag = int(sys.argv[5])
# build_files('trn_ft.txt','trn_lbl_mat.txt','train_X.txt')
vocab = {}
with open(vocab_files,'r',encoding='latin-1') as f:
	for i,v in enumerate(f):
		words,values = v.split(' ')
		vocab[words.encode('latin-1')]=(i,float(values))
total_fts = len(vocab.keys())

print(total_fts)
print(vocab[b'<s>'])


with open(root_a,'r',encoding='latin-1') as ftr,open(root_b,'r',encoding='latin-1') as ltr:
	UNK = vocab[b'UNK']
	ft_mat = open(root_c,'w',encoding='latin-1')
	nr,nc = map(int,ltr.readline().split(' '))

	print("%d %d"%(total_fts,nc),file=ft_mat)
	i=0
	while i<nr:
		count = 0
		line = ftr.readline().strip()
		labels = ltr.readline().strip()
		
		if labels =='':
			continue
		
		sentence = ["%d:%f"%vocab[b'<s>']]
		
		for word in line.split(' '):
			k = b"%s"%(word.encode('latin-1'))
			if k == b'.':
				sentence.append("%d:%f"%vocab[b'<dot>'])
			else:
				idx_word = vocab.get(k,UNK)
				count+=idx_word[1]
				sentence.append("%d:%f"%idx_word)

		sentence.append("%d:%f"%vocab[b'</s>'])
		
		i+=1
		if flag==1 and count==0:
			continue
		
		print(labels+' '+' '.join(sentence),file=ft_mat)
		print('[%d/%d]'%(i,nr),end='\r')
	ft_mat.close()
