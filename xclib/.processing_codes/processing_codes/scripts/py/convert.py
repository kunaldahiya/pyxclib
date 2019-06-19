# Create dataset from raw text
# python3 convert.py features.txt index.txt output.txt "arrow|tab"

from sklearn.feature_extraction.text import TfidfVectorizer
import os
import sys
import numpy as np
import json
import re
import scipy.sparse as sp

text_file = sys.argv[1]
indx_file = sys.argv[2]
outp_file = sys.argv[3]
delimiter = "\t" if sys.argv[4] == "tab" else "->"
def write_sparse_file(labels, filename, header=True):
    if not isinstance(labels, sp.csr_matrix):
        labels=labels.tocsr()
    with open(filename, 'w') as f:
        if header:
            print("%d %d"%(labels.shape[0],labels.shape[1]),file=f)
        for y in labels:
            idx = y.__dict__['indices']
            val = y.__dict__['data']
            sentence = ' '.join(['{}:{}'.format(x,v) for x,v in zip(idx,val)])
            print(sentence,file=f)

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


def read_text(fname,indx):
    text = []
    with open(fname, 'r',encoding='latin') as fp:
        text = fp.readlines()
    text = list(map(lambda x: clean_text(text[x].strip().split(delimiter)[1]),indx))
    return text

def myconverter(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError

def compute_features(text):
    tf = TfidfVectorizer(min_df=3, dtype=np.float32, stop_words='english', norm=None)
    tf.fit(text)
    print("#features: ", len(tf.vocabulary_))
    return tf.transform(text), tf.vocabulary_

if __name__=='__main__':
    print("Reading index")
    indx = []
    with open(indx_file,'r') as f:
        indx = list(map(lambda x: int(x.strip())-1,f))

    print("Reading text and rearranging")
    text = read_text(text_file, indx)
    print("Computing features")
    features, vocab = compute_features(text)

    print("Writing data")
    vocab_ = [None]*len(vocab)
    for k, v in vocab.items():
        vocab_[v] = k

    with open('Xf.txt', 'w') as fp:
        for item in vocab_:
            fp.write(item + "\n")

    write_sparse_file(features,outp_file,header=True)
    print("Thank you!!")