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

def main():
    data = sys.argv[2]
    output = sys.argv[3]
    vocabY = sys.argv[4]
    # text feature object
    t_obj = tfidf(min_df=2,stop_words=[])
    dat_obj = t_obj.fit_transform(open(data,'r',encoding='latin1'))
    num_instances = dat_obj.shape[0]
    dat_obj = sp.hstack([dat_obj,np.zeros((num_instances,1),np.float32)]).tolil()
    data_not_present = np.where(np.ravel(dat_obj.sum(axis=1))==0)[0]
    print("DATA is not present in %d documents"%len(data_not_present))
    dat_obj[data_not_present,-1]=1.0
    vocab = t_obj.get_feature_names()
    data_utils.write_sparse_file(dat_obj.tocsr(),output)
    with open(vocabY,'w') as f:
        for v in vocab:
            print("%s"%(v),file=f)
        print("UNK",file=f)

if __name__ == '__main__':
    main()
