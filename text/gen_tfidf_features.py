# Generate TF-IDF features for a list of documents.
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.corpus import stopwords


def gen_tfidf_features(raw_text,
                       encoding='utf-8',
                       use_stopwords=True,
                       max_df=0.9,
                       min_df=3):
    """
        Generate tf-idf features
    """
    if use_stopwords:
        stop_words = 'english'
    else:
        stop_words = None
    vectorizer = TfidfVectorizer(encoding=encoding,
                                 decode_error='ignore',
                                 stop_words=stop_words,
                                 max_df=max_df,
                                 min_df=min_df,
                                 dtype=np.float32)
    tfidf_features = vectorizer.fit_transform(raw_text)
    return tfidf_features
