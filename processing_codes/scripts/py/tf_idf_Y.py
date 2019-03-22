"""
    Generate vectorized text and tf-idf for given text
"""
import os
import sys
import numpy as np
from xctools.text import text_utils
from xctools.data import data_utils
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
import scipy.sparse as sp
cachedStopWords = []
# cachedStopWords+=stopwords.words("english")
# cachedStopWords+= '\' \" : ; , < > . ~ ` ! @ # ^ & * ( ) + = - _ { } [ ] '.split(' ')

def _tokenizer(s):
    return s.split(" ")

def main():
    data = sys.argv[2]
    output = sys.argv[3]
    vocabY = sys.argv[4]
    with open(data,'r') as f:
        lines = f.readlines()
        empty_lines = []
        for line in range(len(lines)):
            lines[line] = lines[line].strip().lower()
            if lines[line] == '':
                empty_lines.append(line)
                print(line+1)
    print("Total empty lines are %d"%(len(empty_lines)))
    # text feature object
    # t_obj = tfidf(min_df=2, stop_words=[], tokenizer=_tokenizer)
    t_obj = tfidf(min_df=2, stop_words=[])
    dat_obj = t_obj.fit_transform(lines)
    num_instances = dat_obj.shape[0]
    dat_obj = sp.hstack([dat_obj,np.zeros((num_instances,1),np.float32)]).tolil()
    data_not_present = np.where(np.ravel(dat_obj.sum(axis=1))==0)[0]
    print("DATA is not present in %d documents"%len(data_not_present))
    # for data in data_not_present:
    print(dat_obj[empty_lines])
    dat_obj[data_not_present,-1]=1.0
    vocab = t_obj.get_feature_names()
    data_utils.write_sparse_file(dat_obj.tocsr(),output)
    with open(vocabY,'w') as f:
        for v in vocab:
            print("%s"%(v),file=f)
        print("unk",file=f)

if __name__ == '__main__':
    main()
