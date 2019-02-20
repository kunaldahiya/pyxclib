"""
        Text utlities
        - Use sparse matrices which is suitable for large datasets
"""
import os
import re
import pickle
import logging
import json
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, save_npz
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

__author__ = 'KD'

def save_vectorized_text(fname, vectorized_text):
    """
        Save vectorized text as pickle file
    """
    with open(fname, 'wb') as fp:
        pickle.dump(vectorized_text, fp)

def save_vocabulary(fname, vocabulary):
    """
        Save vocabulary as json file
    """
    with open(fname, 'w') as fp:
        json.dump(vocabulary, fp, indent=4)

def save_mat(fname, sp_mat):
    """
        Save sparse matrix
    """
    with open(fname, 'wb') as fp:
        save_npz(fp, sp_mat)


class TFIDF(object):
    """
        Compute TF-IDF features
    """
    def __init__(self, vocabulary_size):
        self.vocabulary_size = vocabulary_size
        self.inv_doc_freq = None

    def fit(self, text):
        """
            Compute inverse document frequency from given data
            Args:
                text: list: vectorized text (i.e. word id's of each document)
        """
        term_freq = self.vectorized_text_to_term_freq(text)
        self.inv_doc_freq = self.compute_inv_doc_freq(term_freq)

    def transform(self, text, use_log=False):
        """
            Compute TF-IDF features for given data
            Args:
                text: list: vectorized text (i.e. word id's of each document)
            Returns:
                csr_matrix: TF-IDF features
        """
        term_freq = self.vectorized_text_to_term_freq(text)
        if use_log:
            term_freq = csr_matrix.log1p(term_freq)
        return self.tf_idf(term_freq)

    def compute_doc_frequency(self, term_freq):
        """
            #documents in which a word appears
            +1 to avoid zeros
            Args:
                term_freq: csr_matrix: term frequency matrix (num_docs, num_words)
            Returns:
                np.array: Document frequency for each word
        """
        return 1 + np.array(term_freq.sum(axis=0)).reshape(self.vocabulary_size)

    def compute_inv_doc_freq(self, term_freq):
        """
            Compute inverse document frequency
            Args:
                term_freq: csr_matrix: term frequency matrix (num_docs, num_words)
            Returns:
                idf: np.array: inverse document frequency
        """
        num_docs, _ = term_freq.shape
        inv_doc_freq = num_docs / self.compute_doc_frequency(term_freq)
        return np.log(inv_doc_freq).reshape(1, self.vocabulary_size)

    def vectorized_text_to_term_freq(self, vectorized_text):
        """
            Convert vectorized text to term frequency matrix
            Args:
                vectorized_text: list: vectorized text (i.e. word id's of each document)
            Returns:
                term_freq: csr_matrix: term frequency matrix
        """
        num_samples = len(vectorized_text)
        term_freq = lil_matrix((num_samples, self.vocabulary_size), dtype=np.float32)
        for idx, item in enumerate(vectorized_text):
            for i in item:
                term_freq[idx, i] += 1
        return term_freq.tocsr()

    def tf_idf(self, term_freq):
        """
            Compute tf-idf features
            Args:
                term_freq: csr_matrix: term frequency matrix
            Returns:
                tf-idf: csr_matrix: tf-idf features
        """
        return term_freq.multiply(self.inv_doc_freq)


class TextUtility(object):
    """
        Text utilities
    """

    def __init__(self,
                 max_df=1.0,
                 min_df=1,
                 vocabulary={},
                 min_word_length=2,
                 max_vocabulary_size=-1,
                 remove_stopwords=True):
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger("TextUtils")
        self.max_df = max_df
        self.min_df = min_df
        self.vocabulary = vocabulary
        self.max_vocabulary_size = max_vocabulary_size
        self.remove_stopwords = remove_stopwords
        self.stop_words = ()
        if self.remove_stopwords:
            self._get_stopwords()
        self.min_word_length = min_word_length

    def _get_stopwords(self, fname='stopwords.txt'):
        if fname != None and os.path.isfile(fname):
            with open(fname) as f:
                stop_words = f.readlines()
            self.stop_words = set([item.rstrip("\n") for item in stop_words])
        else:
            self.stop_words = set(stopwords.words('english'))

    def print_statistics(self):
        """
            Print object statistics
        """
        stats = {
            'min_df': self.min_df,
            'max_df': self.max_df,
            'remove_stopwords': self.remove_stopwords,
            'vocab_size': len(self.vocabulary)
        }
        print(stats)

    def fit(self, textf):
        """
            Fit as per the data
        """
        parsed_text = self._parse_text(textf, is_train=True)
        self.print_statistics()
        count_mat = self._compute_statistics(parsed_text)
        self._process_vocabulary(count_mat)

    def save(self, fname):
        """
            Save the model
        """
        model = {
            'max_df' : self.max_df,
            'min_df' : self.min_df,
            'vocabulary' : self.vocabulary,
            'max_vocabulary_size' : self.max_vocabulary_size,
            'remove_stopwords' : self.remove_stopwords,
            'stop_words' : self.stop_words,
            'min_word_length' : self.min_word_length
        }
        with open(fname, 'wb') as fp:
            pickle.dump(model, fp)

    def restore(self, fname):
        """
            Restore the model
        """
        with open(fname, 'rb') as fp:
            model = pickle.load(fp)
        self.max_df = model['max_df']
        self.min_df = model['min_df']
        self.vocabulary = model['vocabulary']
        self.max_vocabulary_size = model['max_vocabulary_size']
        self.remove_stopwords = model['remove_stopwords']
        self.stop_words = model['stop_words']
        self.min_word_length = model['min_word_length']

    def _clean_text(self, sentence):
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
        return sentence.strip().lower()

    def _parse_text(self, textf, is_train=False, progress_step=50000):
        self.log.info("Parsing given text!")
        with open(textf,encoding='latin') as f:
            text = f.readlines()
        parsed_text = []
        total_documents = len(text)
        for idx, item in enumerate(text):
            if idx % progress_step == 0:
                self.log.info("Parsing progress: [%d/%d]", idx,
                              total_documents)
            item = self._clean_text(item)
            temp = self._tokenize(item)
            parsed_text.append(temp)
            # Assign a unique mapping for each word for train data
            if is_train:
                for item2 in temp:
                    if item2 not in self.vocabulary:
                        self._update_vocabulary(item2)
        del text
        return parsed_text

    def _compute_statistics(self, parsed_text, progress_step=50000):
        self.log.info("Computing statistics for the given data!")
        count_mat = lil_matrix(
            (len(parsed_text), len(self.vocabulary)), dtype=np.int)
        total_documents = len(parsed_text)
        for idx, sentence in enumerate(parsed_text):
            if idx % progress_step == 0:
                self.log.info("Statistics progress: [%d/%d]", idx,
                              total_documents)
            for token in sentence:
                count_mat[idx, self.vocabulary[token]] += 1
        return count_mat.tocsr()

    def _tokenize(self, sentence):
        """
            Ignores words not in vocabulary
        """
        words = word_tokenize(sentence.rstrip("\n"))
        words_ = []
        for item in words:
            if len(item) >= self.min_word_length:
                if not self.remove_stopwords or (self.remove_stopwords and
                                                 item not in self.stop_words):
                    words_.append(item)
        if not words_:
            words_.append('<UNK>')
        return words_

    def _update_vocabulary(self, word):
        self.vocabulary[word] = len(self.vocabulary)

    def sort_vocabulary(self):
        """
            Sort vocabulary in lexiographic order
        """
        keys = list(self.vocabulary.keys())
        keys.sort()
        keys_ = ['<PAD>', '<UNK>']
        keys.remove('<UNK>')
        keys_.extend(keys)
        keys = keys_
        self.vocabulary = dict(zip(keys, range(len(keys))))

    def _vectorize(self, parsed_sentence):
        vector = []
        for token in parsed_sentence:
            try:
                idx = self.vocabulary[token]
                vector.append(idx)
            except KeyError:
                print("Key error for :", token)
        return vector

    def _process_vocabulary(self, count_mat):
        count_mat[count_mat != 0] = 1
        doc_freq = np.array(count_mat.sum(axis=0)).reshape(
            len(self.vocabulary))
        num_docs = count_mat.shape[0]
        updated_vocabulary = {}
        idx = 0
        for key, val in self.vocabulary.items():
            if doc_freq[val] // num_docs <= self.max_df and doc_freq[val] >= self.min_df:
                updated_vocabulary[key] = idx
                idx += 1
            else:
                self.stop_words.add(key)
        self.vocabulary = updated_vocabulary
        self.sort_vocabulary()

    def transform(self, textf):
        """
            Transfrom given text data in word vectors
        """
        parsed_text = self._parse_text(textf)
        vectorized_text = []
        for item in parsed_text:
            vectorized_text.append(self._vectorize(item))
        return vectorized_text
