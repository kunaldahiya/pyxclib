import logging
import scipy.sparse as sparse
import _pickle as pickle
import os
import numpy as np
import _pickle as pickle


class BaseClassifier(object):
    def __init__(self, verbose=0, use_bias=True, use_sparse=False):
        self.verbose = verbose
        self.num_labels = None
        self.use_sparse = use_sparse
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('XC-Classifier')
        self.weight = None
        self.bias = None

    def _save_state(self, model_dir, epoch):
        fname = os.path.join(model_dir, 'model_state_{}.pkl'.format(epoch))
        pickle.dump({'weight': self.weight.transpose(), 'bias': self.bias.transpose()},
                    open(fname, 'wb'))

    def _compute_clf_size(self):
        _size = 0
        if self.weight is not None:
            if isinstance(self.weight, np.ndarray):
                _size += self.weight.size*4/(1024*1024*1024)
            else:
                _size += self.weight.nnz*4/(1024*1024*1024)
        else:
            raise AssertionError("Classifier is not yet trained!")
        if self.bias is not None:
            if isinstance(self.bias, np.ndarray):
                _size += self.bias.size*4/(1024*1024*1024)
            else:
                _size += self.bias.nnz*4/(1024*1024*1024)
        return _size

    def save(self, fname):
        pickle.dump({'weight': self.weight.transpose(),
                     'bias': self.bias.transpose(),
                     'use_sparse': self.use_sparse,
                     'num_labels': self.num_labels},
                    open(fname, 'wb'))

    def load(self, fname):
        temp = pickle.load(open(fname, 'rb'))
        self.bias = temp['bias']
        self.weight = temp['weight']
        self.use_sparse = temp['use_sparse']
        self.num_labels = temp['num_labels']

    def __repr__(self):
        return "num_labels: {}, use_sparse: {}".format(self.num_labels, self.use_sparse)
