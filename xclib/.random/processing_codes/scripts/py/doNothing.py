from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from xctools.data import data_utils as du
import numpy as np
import sys
# 
cachedStopWords = []
# cachedStopWords+=stopwords.words("english")
# cachedStopWords+= '\' \" : ; , < > . ~ ` ! @ # ^ & * ( ) + = - _ { } [ ] '.split(' ')

root_a = sys.argv[1]
root_b = sys.argv[2]
root_c = sys.argv[3]

