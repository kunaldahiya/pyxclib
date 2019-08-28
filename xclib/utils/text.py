"""
    Text utlities
    - Use sparse matrices which is suitable for large datasets
"""
import os
import re
import json
import _pickle as pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, VectorizerMixin
from functools import partial
import numbers
from collections import defaultdict


__author__ = 'KD'


def clean_sent(line, max_len=None):
    """Clean sentence
    Arguments
    ---------
    line: str
        line to clean
    max_len: int or None, optional, default=None
        cutoff length of a sentence
    """
    line = re.sub(
        r"[^A-Za-z0-9(),!?'`.#\@\$\%\^\&\*\{\}\+\-\|\\]", " ", line)
    line = re.sub(
        r"([A-Za-z0-9(),!?'`.#\@\$\%\^\&\*\{\}\+\-\|\\])+?[.]+(\s|$)", r"\1 /end/ ", line)
    line = re.sub(r"\s", " ", line)
    line = re.sub(r"\'s", " ", line)
    line = re.sub(r"\'ve", " ", line)
    line = re.sub(r"\'re", " ", line)
    line = re.sub(r"\'ll", " ", line)
    line = re.sub(r"\'", r" \' ", line)
    line = re.sub(r",", " , ", line)
    line = re.sub(r"!", " ! ", line)
    line = re.sub(r"\(", r" \( ", line)
    line = re.sub(r"\)", r" \) ", line)
    line = re.sub(r"\?", r" \? ", line)
    line = re.sub(r"\s\s+", r" ", line)
    line = re.sub(r"/end/", r"", line)
    line = re.sub(r"\b(.)\1+\b", r"\1\1", line)
    if max_len is not None:
        return ' '.join(line.lower().split(' ')[:max_len])
    else:
        return line.lower()


#  Source: https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/feature_extraction/text.py
class BoWFeatures(TfidfVectorizer):
    """Convert a collection of raw documents to a matrix of TF-IDF features.

    Parameters
    ----------
    input : string {'filename', 'file', 'content'}
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.
        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.
        Otherwise the input is expected to be the sequence strings or
        bytes items are expected to be analyzed directly.
    encoding : string, 'latin1' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.
    decode_error : {'strict', 'ignore', 'replace'} (default='strict')
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.
    strip_accents : {'ascii', 'unicode', None} (default=None)
        Remove accents and perform other character normalization
        during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.
        Both 'ascii' and 'unicode' use NFKD normalization from
        :func:`unicodedata.normalize`.
    lowercase : boolean (default=True)
        Convert all characters to lowercase before tokenizing.
    preprocessor : callable or None or custom (default=None)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.
        Custom is defined above
    tokenizer : callable or None (default=None)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
        Only applies if ``analyzer == 'word'``.
    analyzer : string, {'word', 'char', 'char_wb'} or callable
        Whether the feature should be made of word or character n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries; n-grams at the edges of words are padded with space.
        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.
    stop_words : string {'english'}, list, or None (default=english)
        If a string, it is passed to _check_stop_list and the appropriate stop
        list is returned. 'english' is currently the only supported string
        value.
        There are several known issues with 'english' and you should
        consider an alternative (see :ref:`stop_words`).
        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.
        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.
    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp selects tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).
    ngram_range : tuple (min_n, max_n) (default=(1, 1))
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.
    max_df : float in range [0.0, 1.0] or int (default=1.0)
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.
    min_df : float in range [0.0, 1.0] or int (default=1)
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.
    max_features : int or None (default=None)
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.
        This parameter is ignored if vocabulary is not None.
    vocabulary : Mapping or iterable, optional (default=None)
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents.
    binary : boolean (default=False)
        If True, all non-zero term counts are set to 1. This does not mean
        outputs will have only 0/1 values, only that the tf term in tf-idf
        is binary. (Set idf and normalization to False to get 0/1 outputs.)
    dtype : type, optional (default=float32)
        Type of the matrix returned by fit_transform() or transform().
    norm : 'l1', 'l2' or None, optional (default='l2')
        Each output row will have unit norm, either:
        * 'l2': Sum of squares of vector elements is 1. The cosine
        similarity between two vectors is their dot product when l2 norm has
        been applied.
        * 'l1': Sum of absolute values of vector elements is 1.
        See :func:`preprocessing.normalize`
    use_idf : boolean (default=True)
        Enable inverse-document-frequency reweighting.
    smooth_idf : boolean (default=True)
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.
    sublinear_tf : boolean (default=False)
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    Attributes
    ----------
    vocabulary_ : dict
        A mapping of terms to feature indices.
    idf_ : array, shape (n_features)
        The inverse document frequency (IDF) vector; only defined
        if ``use_idf`` is True.
    stop_words_ : set
        Terms that were ignored because they either:
          - occurred in too many documents (`max_df`)
          - occurred in too few documents (`min_df`)
          - were cut off by feature selection (`max_features`).
        This is only available if no vocabulary was given.
    """

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor='custom',
                 tokenizer=None, analyzer='word',
                 stop_words='english', token_pattern='(?u)\b\w\w+\b',
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.float32, norm='l2', use_idf=True,
                 smooth_idf=True, sublinear_tf=False, max_len=None):
        if preprocessor == 'custom':
            preprocessor = self.custom_preprocessor(max_len)
        super().__init__(
            input, encoding, decode_error, strip_accents, lowercase, 
            preprocessor, tokenizer, analyzer, stop_words, token_pattern,
            ngram_range, max_df, min_df, max_features, vocabulary, binary,
            dtype, norm, use_idf, smooth_idf, sublinear_tf)

    def custom_preprocessor(self, max_len):
        return partial(clean_sent, max_len=max_len)

    def stats(self):
        stats = {
            'min_df': self.min_df,
            'max_df': self.max_df,
            'vocab_size': len(self.vocabulary_)
        }
        return stats

    def load(self, fname):
        with open(fname, 'wb') as fp:
            pickle.dump(self.__dict__, fp)

    def save(self, fname):
        self = pickle.load(fname, 'rb')


class SeqFeatures(VectorizerMixin):
    """
        Tokenize text as a sequence
    Parameters
    ----------
    input : string {'filename', 'file', 'content'}
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.
        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.
        Otherwise the input is expected to be the sequence strings or
        bytes items are expected to be analyzed directly.
    encoding : string, 'latin1' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.
    decode_error : {'strict', 'ignore', 'replace'} (default='strict')
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.
    strip_accents : {'ascii', 'unicode', None} (default=None)
        Remove accents and perform other character normalization
        during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.
        Both 'ascii' and 'unicode' use NFKD normalization from
        :func:`unicodedata.normalize`.
    lowercase : boolean (default=True)
        Convert all characters to lowercase before tokenizing.
    preprocessor : callable or None or custom (default=None)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.
        Custom is defined above
    tokenizer : callable or None (default=None)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
        Only applies if ``analyzer == 'word'``.
    analyzer : string, {'word', 'char', 'char_wb'} or callable
        Whether the feature should be made of word or character n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries; n-grams at the edges of words are padded with space.
        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.
    stop_words : string {'english'}, list, or None (default=english)
        If a string, it is passed to _check_stop_list and the appropriate stop
        list is returned. 'english' is currently the only supported string
        value.
        There are several known issues with 'english' and you should
        consider an alternative (see :ref:`stop_words`).
        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.
        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.
    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp selects tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).
    ngram_range : tuple (min_n, max_n) (default=(1, 1))
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.
    max_df : float in range [0.0, 1.0] or int (default=1.0)
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.
    min_df : float in range [0.0, 1.0] or int (default=1)
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.
    max_features : int or None (default=None)
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.
        This parameter is ignored if vocabulary is not None.
    vocabulary : Mapping or iterable, optional (default=None)
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents.
    max_len : int, optional (default=None)
        Cutoff length for documents
    default_tokens : dict, optional
        default: default={'<PAD>': 0, '<UNK>': 1, '<S>': 2, '</S>': 3}
        Add these tokens manually
    Attributes
    ----------
    vocabulary_ : dict
        A mapping of terms to feature indices.
    stop_words_ : set
        Terms that were ignored because they either:
          - occurred in too many documents (`max_df`)
          - occurred in too few documents (`min_df`)
          - were cut off by feature selection (`max_features`).
        This is only available if no vocabulary was given.

    """

    def __init__(self, input='content', encoding='latin1',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor='custom',
                 tokenizer=None, analyzer='word', ngram_range=(1, 1),
                 stop_words='english', token_pattern=r"(?u)\b\w\w+\b",
                 max_df=1.0, min_df=1, max_features=None, 
                 vocabulary=None, max_len=None, 
                 default_tokens={'<PAD>': 0, '<UNK>': 1, '<S>': 2, '</S>': 3}):
        #TODO: UNK token and PAD token
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.max_df = max_df
        self.min_df = min_df
        self.default_tokens = default_tokens
        if max_df < 0 or min_df < 0:
            raise ValueError("negative value for max_df or min_df")
        self.max_features = max_features
        if max_features is not None:
            if (not isinstance(max_features, numbers.Integral) or
                    max_features <= 0):
                raise ValueError(
                    "max_features=%r, neither a positive integer nor None"
                    % max_features)
        self.vocabulary = vocabulary

    def _create_vocab(self, raw_documents):
        """Create vocabulary where fixed_vocab=False
        """
        vocabulary = defaultdict()
        vocabulary.default_factory = vocabulary.__len__

        analyze = self.build_analyzer()
        for doc in raw_documents:
            feature_counter = {}
            for feature in analyze(doc):
                feature_idx = vocabulary[feature]
                if feature_idx not in feature_counter:
                    feature_counter[feature_idx] = 1
                else:
                    feature_counter[feature_idx] += 1

        vocabulary = dict(vocabulary)
        if not vocabulary:
            raise ValueError("empty vocabulary; perhaps the documents only"
                                " contain stop words")
        return vocabulary

    def _sort_features(self, vocabulary):
        """Sort features by name
        Modifies the vocabulary in place
        """
        sorted_features = sorted(vocabulary.items())
        for new_val, (term, _) in enumerate(sorted_features):
            vocabulary[term] = new_val
        return vocabulary

    def build_analyzer(self):
        """Return a callable that handles preprocessing and tokenization"""
        if callable(self.analyzer):
            return lambda doc: self.analyzer(self.decode(doc))

        preprocess = self.build_preprocessor()

        if self.analyzer == 'char':
            return lambda doc: self._char_ngrams(preprocess(self.decode(doc)))

        elif self.analyzer == 'char_wb':
            return lambda doc: self._char_wb_ngrams(
                preprocess(self.decode(doc)))

        elif self.analyzer == 'word':
            stop_words = self.get_stop_words()
            tokenize = self.build_tokenizer()
            self._check_stop_words_consistency(stop_words, preprocess,
                                               tokenize)
            return lambda doc: self._word_ngrams(
                tokenize(preprocess(self.decode(doc))), stop_words)

        else:
            raise ValueError('%s is not a valid tokenization scheme/analyzer' %
                             self.analyzer)
   
    def add_default_tokens(self, vocabulary):
        # Assumes indices for new tokens are fine
        offset = len(self.default_tokens)
        for key, val in vocabulary.items():
            vocabulary[key] = val+offset

        return {**self.default_tokens, **vocabulary}

    def _limit_features(self, vocabulary, doc_freq):
        """Remove too rare or too common features.
        Prune features that are non zero in more samples than high or less
        documents than low, modifying the vocabulary, and restricting it to
        at most the limit most frequent.
        """
        print("Limit features not working as of now!")
        return vocabulary, set()
        # Calculate a mask based on document frequencies
        if self.max_df is None and self.min_df is None and self.max_features is None:
            return vocabulary, set()

        mask = np.ones(len(doc_freq), dtype=bool)

        if self.max_df is not None:
            mask &= doc_freq <= self.max_df
        if self.min_df is not None:
            mask &= doc_freq >= self.max_df

        if self.max_features is not None and mask.sum() > self.max_features:
            mask_inds = (-doc_freq[mask]).argsort()[:self.max_features]
            new_mask = np.zeros(len(doc_freq), dtype=bool)
            new_mask[np.where(mask)[0][mask_inds]] = True
            mask = new_mask

        new_indices = np.cumsum(mask) - 1  # maps old indices to new
        removed_terms = set()
        for term, old_index in list(vocabulary.items()):
            if mask[old_index]:
                vocabulary[term] = new_indices[old_index]
            else:
                del vocabulary[term]
                removed_terms.add(term)
        kept_indices = np.where(mask)[0]
        if len(kept_indices) == 0:
            raise ValueError("After pruning, no terms remain. Try a lower"
                             " min_df or a higher max_df.")
        return vocabulary, removed_terms

    def print_statistics(self):
        """
            Print object statistics
        """
        stats = {
            'min_df': self.min_df,
            'max_df': self.max_df,
            'vocab_size': len(self.vocabulary_)
        }
        print(stats)

    def build_token_mapper(self):
        return lambda x: self.vocabulary_[x]

    def _tokenize(self, text):
        token_mapper = self.build_token_mapper()
        analyze = self.build_analyzer()
        for doc in text:
            _out = ['<S>'] + analyze(doc) + ['</S>']
            yield list(map(token_mapper, _out))

    def fit(self, text):
        if isinstance(text, str):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        if self.vocabulary is None:
            vocabulary = self._create_vocab(text)
            vocabulary = self._sort_features(vocabulary)
            vocabulary, stop_words = self._limit_features(vocabulary, None)
            vocabulary = self.add_default_tokens(vocabulary)
            self.stop_words_ = stop_words
            self.vocabulary_ = vocabulary

    def transform(self, text):
        return list(self._tokenize(text))

    def load(self, fname):
        with open(fname, 'wb') as fp:
            pickle.dump(self.__dict__, fp)

    def save(self, fname):
        self = pickle.load(fname, 'rb')
