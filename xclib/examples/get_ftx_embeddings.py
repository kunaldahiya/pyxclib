'''
run as: python3 get_ftx_embeddings.py <fasttext_model> <vocabulary_file> <output_file>
example: python3 get_ftx_embeddings.py wiki.en.bin Xf.txt fasttextB_embeddings_300d.npy

Compute fasttext embeddings for a given vovabulary file
    * input file format: a token in each line
    * pre-trained fasttext model is availabe at: 
    https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
'''

import sys
import re
from xclib.embeddings.fasttext_embeddings import load_model, get_vectors
import numpy as np


def spell_out_numbers(sentence):
    sentence = re.sub(r"0", " zero", sentence)
    sentence = re.sub(r"1", " one", sentence)
    sentence = re.sub(r"2", " two", sentence)
    sentence = re.sub(r"3", " three", sentence)
    sentence = re.sub(r"4", " four", sentence)
    sentence = re.sub(r"5", " five", sentence)
    sentence = re.sub(r"6", " six", sentence)
    sentence = re.sub(r"7", " seven", sentence)
    sentence = re.sub(r"8", " eight", sentence)
    sentence = re.sub(r"9", " nine", sentence)
    return sentence.lstrip()


def preprocess(line):
    line = line.rstrip()
    line = spell_out_numbers(line)
    return re.sub(r" ", "_", line)


def load_vocabulary(infile):
    with open(infile, 'r',encoding='latin-1') as fp:
        vocabulary = fp.readlines()
    return [preprocess(item) for item in vocabulary]


if __name__ == '__main__':
    model = load_model(sys.argv[1])
    vocabulary = load_vocabulary(sys.argv[2])
    embeddings = get_vectors(model, vocabulary)
    np.save(
        sys.argv[3],
        embeddings
    )
