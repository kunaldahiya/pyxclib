"""
    Generate vectorized text and tf-idf for given text
"""
import os
import sys
import numpy as np
from xclib.utils import text


def compute_features():
    corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document document?']

    # Compute BoW features as
    obj = text.BoWFeatures()
    obj.fit(corpus)
    print(obj.transform(corpus).toarray())
