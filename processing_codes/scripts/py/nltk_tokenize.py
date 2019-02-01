import nltk
import sys
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
cachedStopWords = []
cachedStopWords+=stopwords.words("english") 
# cachedStopWords+= '\' \" : ; , < > . ~ ` ! @ # ^ & * ( ) + = - _ { } [ ] '.split(' ')

lemmatizer = WordNetLemmatizer()
porter = nltk.PorterStemmer()

raw_text = sys.argv[1]
processed_text = sys.argv[2]
to = open(processed_text,'w',encoding='latin-1')

def clean_text(sentence):
    sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)
    sentence = re.sub(r"\'s ", " ", sentence)
    sentence = re.sub(r"\'ve ", " ", sentence)
    sentence = re.sub(r"\'re ", " ", sentence)
    sentence = re.sub(r"\'ll ", " ", sentence)
    sentence = re.sub(r"\'", " \' ", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\(", " \( ", sentence)
    sentence = re.sub(r"\)", " \) ", sentence)
    sentence = re.sub(r"\?", " \? ", sentence)
    sentence = re.sub(r"\s{2,}", " ", sentence)
    #sentence = re.sub(r'\d+', '', sentence)
    return sentence.lower()


def word_wise():
	with open(raw_text,'r',encoding='latin-1') as f:
		for sentence in f:
			words = [porter.stem(lemmatizer.lemmatize(w)) for w in nltk.word_tokenize(sentence.strip()) if (w not in cachedStopWords) and len(w)>1]
			print(' '.join(words),file=to)

def sentence_wise():
	with open(raw_text,'r',encoding='latin-1') as f:
		for sentences in f:
			paragraph = []
			for sentence in nltk.sent_tokenize(sentences.strip()):
				words = [porter.stem(lemmatizer.lemmatize(w)) for w in nltk.word_tokenize(sentence.strip()) if (w not in cachedStopWords) and len(w)>1]
				paragraph.append(' '.join(words))
			print(' . '.join(paragraph),file=to)

def sentence_wise_no_tokenise():
	with open(raw_text,'r',encoding='latin-1') as f:
		for sentences in f:
			print(clean_text(sentences.strip()), file=to)


sentence_wise_no_tokenise()
