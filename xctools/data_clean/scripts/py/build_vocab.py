import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
# 
cachedStopWords = []
# cachedStopWords+=stopwords.words("english") 
# cachedStopWords+= '\' \" : ; , < > . ~ ` ! @ # ^ & * ( ) + = - _ { } [ ] '.split(' ')

root = sys.argv[1]
min_df = int(sys.argv[2])

print(root+'temp_X.txt')
print(min_df)
vocab_file = open(root+'VOCAB.txt','w',encoding='latin-1')

with open(root+'temp_X.txt','r',encoding='latin-1') as f:
	vectorizer = TfidfVectorizer(min_df=min_df,encoding='latin-1',stop_words=cachedStopWords)
	vectorizer.fit_transform(f)
	idf = vectorizer.idf_
	vectorizer.vocabulary_.pop('.',None)
	vocabs = list(vectorizer.vocabulary_.keys())
	vocabs.sort()
	for i,key in enumerate(vocabs):
		print(f'{key} {idf[vectorizer.vocabulary_[key]]}',file=vocab_file)

	for extra in ['UNK','</s>','<s>']:
		print(f'{extra} 0.0',file=vocab_file)

vocab_file.close()
total_fts = i+4
print(total_fts)
