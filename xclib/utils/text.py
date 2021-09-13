"""
    Text utlities
    - Use sparse matrices which is suitable for large datasets
"""
import re
import array
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    from sklearn.feature_extraction.text import _VectorizerMixin as VectorizerMixin
except: # for sklearn < 0.23
    from sklearn.feature_extraction.text import VectorizerMixin
from functools import partial
import numbers
from sklearn.preprocessing import normalize
from collections import defaultdict
import scipy.sparse as sp


__author__ = 'KD'


def clean_sent(line, pattern_keep, pattern_end, pattern_pad,
               pattern_remove, max_len=None):
    """Clean sentence
    Arguments
    ---------
    line: str
        line to clean
    max_len: int or None, optional, default=None
        cutoff length of a sentence
    """
    line = pattern_keep.sub(" ", line)
    line = pattern_end.sub(r"\1 /end/ ", line)
    line = pattern_remove.sub(" ", line)
    line = pattern_pad.sub(r" \1 ", line)
    line = re.sub(r"\b(.)\1+\b", r"\1\1", line)
    if max_len is not None:
        return ' '.join(line.lower().split(' ')[:max_len])
    else:
        return line.lower()


class BaseExtractor(VectorizerMixin):
    """
        Base class for feature extrator
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

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor='custom',
                 tokenizer=None, analyzer='word', ngram_range=(1, 1),
                 stop_words='english', token_pattern=r"(?u)\b\w\w+\b",
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, max_len=None):
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
        self.max_len = max_len

    def _create_vocab(self, raw_documents, vocabulary=None):
        """Create vocabulary where fixed_vocab=False
        """
        if vocabulary is None:
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        analyze = self.build_analyzer()
        df = {}
        tf = {}
        for doc in raw_documents:
            temp = analyze(doc)
            for feature in temp:
                feature_idx = vocabulary[feature]
                if feature_idx not in tf:
                    tf[feature_idx] = 1
                else:
                    tf[feature_idx] += 1

            for feature in set(temp):
                feature_idx = vocabulary[feature]
                if feature_idx not in df:
                    df[feature_idx] = 1
                else:
                    df[feature_idx] += 1

        if not vocabulary:
            raise ValueError("empty vocabulary; perhaps the documents only"
                             " contain stop words")
        else:
            vocabulary = dict(vocabulary)
        return vocabulary, df, tf

    def build_preprocessor(self):
        if self.preprocessor == 'custom':
            pattern_keep = re.compile(
                r"[^A-Za-z0-9(),!?'`.#\@\$\%\^\&\*\{\}\+\-\|\\]")
            pattern_pad = re.compile(r"([',!?()])")
            pattern_remove = re.compile(r"\'s|\'ve|\'re|\'ll|\s\s+|/end/")
            pattern_end = re.compile(
                r"([A-Za-z0-9(),!?'`.#\@\$\%\^\&\*\{\}\+\-\|\\])+?[.]+(\s|$)")   
            self.preprocessor = partial(
                clean_sent, max_len=self.max_len,
                pattern_keep=pattern_keep, pattern_end=pattern_end,
                pattern_pad=pattern_pad, pattern_remove=pattern_remove)
        return super().build_preprocessor()

    def _sort_features(self, vocabulary, df, tf):
        """Sort features by name
        Modifies the vocabulary in place
        """
        _tf = [None]*len(tf)
        _df = [None]*len(df)
        sorted_features = sorted(vocabulary.items())
        for new_val, (term, old_val) in enumerate(sorted_features):
            vocabulary[term] = new_val
            _tf[new_val] = tf[old_val]
            _df[new_val] = df[old_val]
        return vocabulary, _df, _tf

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

    def _limit_features(self, vocabulary, df, tf, high, low):
        """Remove too rare or too common features.
        Prune features that are non zero in more samples than high or less
        documents than low, modifying the vocabulary, and restricting it to
        at most the limit most frequent.
        """
        # Calculate a mask based on document frequencies
        df = np.array(df, dtype=np.int32)
        tf = np.array(tf, dtype=np.float32)

        if high is None and low is None and self.max_features is None:
            return vocabulary, set()

        mask = np.ones(len(tf), dtype=bool)

        if high is not None:
            mask &= df <= high
        if low is not None:
            mask &= df >= low

        if self.max_features is not None and mask.sum() > self.max_features:
            mask_inds = (-tf[mask]).argsort()[:self.max_features]
            new_mask = np.zeros(len(tf), dtype=bool)
            new_mask[np.where(mask)[0][mask_inds]] = True
            mask = new_mask
        new_indices = np.cumsum(mask) - 1  # maps old indices to new
        removed_terms = set()
        _ind_map = {}
        for term, old_index in list(vocabulary.items()):
            if mask[old_index]:
                vocabulary[term] = new_indices[old_index]
                _ind_map[new_indices[old_index]] = old_index
            else:
                del vocabulary[term]
                removed_terms.add(term)
        _df, _tf = [None]*len(vocabulary), [None]*len(vocabulary)
        for key, val in _ind_map.items():
            _df[key] = df[val]
            _tf[key] = tf[val]
        if len(vocabulary) == 0:
            raise ValueError("After pruning, no terms remain. Try a lower"
                             " min_df or a higher max_df.")
        return vocabulary, removed_terms, _df, _tf

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

    def load(self, fname):
        with open(fname, 'wb') as fp:
            pickle.dump(self.__dict__, fp)

    def save(self, fname):
        self = pickle.load(fname, 'rb')


def dict_to_list(x):
    """Assumes keys to be positive int"""
    _x = [0] * (max(x.keys()) +1)
    for k, v in x.items():
        _x[k] = v
    return _x


#  Source: https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/feature_extraction/text.py
class BoWFeatures(BaseExtractor):
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
                 stop_words='english', token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.float32, norm='l2', use_idf=True,
                 smooth_idf=True, sublinear_tf=False, max_len=None):
        super().__init__(
            input, encoding, decode_error, strip_accents,
            lowercase, preprocessor, tokenizer, analyzer, ngram_range,
            stop_words, token_pattern, max_df, min_df, max_features,
            vocabulary, max_len)
        self.binary = binary
        self.dtype = dtype
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.idf = None
        self.norm = norm
        self.sublinear_tf = sublinear_tf

    def fit(self, raw_documents):
        if self.vocabulary is None:
            vocabulary, df, tf = self._create_vocab(raw_documents)
            vocabulary, df, tf = self._sort_features(vocabulary, df, tf)
            high = (self.max_df
                    if isinstance(self.max_df, numbers.Integral)
                    else self.max_df * len(raw_documents))
            low = (self.min_df
                   if isinstance(self.min_df, numbers.Integral)
                   else self.min_df * len(raw_documents))
            if high < low:
                raise ValueError(
                    "max_df corresponds to < documents than min_df")
            vocabulary, stop_words, df, tf = self._limit_features(
                vocabulary, df, tf, high, low)
            self.stop_words_ = stop_words
            self.vocabulary_ = vocabulary
            if self.use_idf:
                self.idf = self._compute_idf(df, len(raw_documents))
        else:
            if self.use_idf:
                _, df, _ = self._create_vocab(raw_documents, self.vocabulary)
                self.idf = self._compute_idf(
                    dict_to_list(df), len(raw_documents))
            self.vocabulary_ = self.vocabulary

    def transform(self, raw_documents, num_threads=1):
        X = self._compute_countf(raw_documents)

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

        if self.use_idf:
            X = X * self.idf

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X

    def fit_transform(self):
        pass

    def _compute_countf(self, raw_documents):
        """Create sparse feature matrix
        """
        def _make_int_array():
            """Construct an array.array of a type suitable
            for scipy.sparse indices.
            """
            return array.array(str("i"))
        analyze = self.build_analyzer()
        j_indices = []
        indptr = []

        values = _make_int_array()
        indptr.append(0)
        for doc in raw_documents:
            feature_counter = {}
            for feature in analyze(doc):
                try:
                    feature_idx = self.vocabulary_[feature]
                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx] = 1
                    else:
                        feature_counter[feature_idx] += 1
                except KeyError:
                    # Ignore out-of-vocabulary items
                    continue

            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))

        if indptr[-1] > 2147483648:  # = 2**31 - 1
            if _IS_32BIT:
                raise ValueError(('sparse CSR array has {} non-zero '
                                  'elements and requires 64 bit indexing, '
                                  'which is unsupported with 32 bit Python.')
                                 .format(indptr[-1]))
            indices_dtype = np.int64

        else:
            indices_dtype = np.int32
        j_indices = np.asarray(j_indices, dtype=indices_dtype)
        indptr = np.asarray(indptr, dtype=indices_dtype)
        values = np.frombuffer(values, dtype=np.intc)

        X = sp.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr) - 1, len(self.vocabulary_)),
                          dtype=self.dtype)
        X.sort_indices()
        return X

    def _compute_idf(self, df, num_samples):
        df = np.array(df, dtype=np.int32)
        num_features = len(df)
        df += int(self.smooth_idf)
        num_samples += int(self.smooth_idf)
        idf = np.log(num_samples / df) + 1
        return sp.diags(idf, offsets=0,
                        shape=(num_features, num_features),
                        format='csr',
                        dtype=self.dtype)


class SeqFeatures(BaseExtractor):
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
        super().__init__(
            input, encoding, decode_error, strip_accents,
            lowercase, preprocessor, tokenizer, analyzer, ngram_range,
            stop_words, token_pattern, max_df, min_df, max_features,
            vocabulary, max_len)
        self.default_tokens = default_tokens

    def add_default_tokens(self, vocabulary):
        # Assumes indices for new tokens are fine
        offset = len(self.default_tokens)
        for key, val in vocabulary.items():
            vocabulary[key] = val+offset

        return {**self.default_tokens, **vocabulary}

    def build_token_mapper(self, x):
        return self.vocabulary_[x] \
            if x in self.vocabulary_ \
            else self.vocabulary_['<UNK>']

    def _tokenize(self, text):
        token_mapper = self.build_token_mapper
        analyze = self.build_analyzer()
        for doc in text:
            _out = ['<S>'] + analyze(doc) + ['</S>']
            yield list(map(token_mapper, _out))

    def fit(self, raw_documents):
        if isinstance(raw_documents, str):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        if self.vocabulary is None:
            vocabulary, df, tf = self._create_vocab(raw_documents)
            vocabulary, df, tf = self._sort_features(vocabulary, df, tf)
            high = (self.max_df
                    if isinstance(self.max_df, numbers.Integral)
                    else self.max_df * len(raw_documents))
            low = (self.min_df
                   if isinstance(self.min_df, numbers.Integral)
                   else self.min_df * len(raw_documents))
            if high < low:
                raise ValueError(
                    "max_df corresponds to < documents than min_df")
            vocabulary, stop_words, _, _ = self._limit_features(
                vocabulary, df, tf, high, low)
            vocabulary = self.add_default_tokens(vocabulary)
            self.stop_words_ = stop_words
            self.vocabulary_ = vocabulary
        else:
            self.vocabulary_ = self.vocabulary

    def transform(self, raw_documents):
        return list(self._tokenize(raw_documents))
