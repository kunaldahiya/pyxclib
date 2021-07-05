import fasttext
import numpy as np


def get_vectors(model, vocabulary):
    """
    Compute fasttext embeddings for a given vovabulary file
    
    Arguments:
    ---------
    model: fasttext.FastText._FastText object
        Fasttext model
    vocabulary: string
        path of input file format which contains a token in each line

    Returns:
        output: np.ndarray
            an array of size V x D
            * V: length of vocabulary
            * D: dimensionality of token vectors
    """
    output = np.zeros(
        (len(vocabulary), model.get_dimension()), dtype='float32')
    for i, v in enumerate(vocabulary):
        output[i] = model.get_word_vector(v)
    return output


def load_model(fname):
    """
    Compute fasttext embeddings for a given vovabulary file
    
    Arguments:
    ---------
    fname: string
        path of input file (.bin) which contains the fasttext model
        Download from:
        * https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
        * https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz

    Returns:
        fasttext.FastText._FastText object
            Fasttext model
    """
    # Return fasttext model
    return fasttext.load_model(fname)
