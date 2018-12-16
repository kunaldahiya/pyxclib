import re
import nltk
from nltk.corpus import stopwords
pattern = r"""(?:[A-Z]\.)+|\d+(?:\.\d+)?%?|\w+(?:[-']\w+)*|(?:[+/\-@&*])"""
vocab = {}
labels = {}
titles = {}
stop = {}
ft_freq = {}

for w in stopwords.words('english'):
	stop[w] = 1
porter = nltk.PorterStemmer()

with open('all_titles.txt','r') as f:
	for line in f:
		x = line.strip().split('\t')
		titles[x[0]] = x[1]
fp = open('temp.txt','w')

with open('raw_data.txt','r') as f:
	for line in f:
		x = line.strip().split(';\t')
		fp.write('%s;\t%s;\t'%(x[0],titles[x[0]]))
		d = x[1]
		d = nltk.regexp_tokenize(d,pattern)
		temp = {}
		for w in d:
			w = w.lower()
			if(len(w)>1 and not stop.has_key(w)):
				w = porter.stem(w)
				if temp.has_key(w):
					temp[w] += 1
				else:
					temp[w] = 1
				vocab[w] = 1
		
		if(titles[x[0]] not in 'NA'):
			d = titles[x[0]]
			d = nltk.regexp_tokenize(d,pattern)
			for w in d:
				w = w.lower()
				if(len(w)>1 and not stop.has_key(w)):
					w = porter.stem(w)
					if temp.has_key(w):
						temp[w] += 1
					else:
						temp[w] = 1
					vocab[w] = 1
		for t in temp:
			fp.write('%s:%d\t'%(t,temp[t]))
			if ft_freq.has_key(t):
				ft_freq[t] += 1
			else:
				ft_freq[t] = 1

		fp.write(';\t')
		lbls = x[2].split(', ')
		temp = {}
		for l in lbls:
			l = l.strip('\'')
			temp[l] = 1
			labels[l] = 1
		for t in temp:
			fp.write('%s\t'%t)
		fp.write('\n')
fp.close()

min_ft_freq = 5
i = 0
fp = open('features.txt','w')

for key,val in sorted(vocab.items()):
	if ft_freq[key]<5:
		continue
	vocab[key] = i
	fp.write('%d\t%s\n'%(i,key))
	i += 1
fp.close()

i = 0
fp = open('labels.txt','w')
for key,val in sorted(labels.items()):
	if key in ['',' ']:
		continue
	labels[key] = i
	fp.write('%d\t%s\n'%(i,key))
	i += 1
fp.close()

print i