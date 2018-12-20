import numpy as np
import os

def load_embeddings(DATA_DIR = "/home/kd/XC/data/word_embeddings", embed='glove', tokens='6B', embedding_dim=100, mode='quiet'):
    embeddings = {}
    f = open(os.path.join(DATA_DIR, '{}.{}.{}d.txt'.format(embed,tokens,embedding_dim)),'r+')
    if embed != 'glove': # In word2vec and fastext, first line is vocabulary dim
        next(f)
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except ValueError:
            if mode == 'verbose':
                print(word)
        embeddings[word] = coefs
    f.close()
    return embeddings
