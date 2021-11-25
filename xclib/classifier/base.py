import logging
import os
import numpy as np
import pickle
import sys
from operator import itemgetter
import math


class BaseClassifier(object):
    """
    Base classifier for sparse or dense data
    (suitable for large label set)

    Parameters:
    -----------
    verbose: int, optional, default=0
        print progress in svm
    feature_type: str, optional, default='sparse'
        feature type: sparse or dense
    use_bias: boolean, optional, default=True
        train bias parameter or not
    """

    def __init__(self, verbose=0, use_bias=True, feature_type='sparse'):
        assert use_bias is True, "Not yet implemented for no bias!"
        self.verbose = verbose
        self.num_labels = None
        self.use_bias = use_bias
        self.feature_type = feature_type
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)
        self.logger = logging.getLogger('XC-Classifier')
        self.weight = None
        self.bias = None

    def _save_state(self, model_dir, epoch):
        fname = os.path.join(model_dir, 'model_state_{}.pkl'.format(epoch))
        pickle.dump({self.__dict__}, open(fname, 'wb'))

    @property
    def model_size(self):
        _size = 0
        if self.weight is not None:
            if isinstance(self.weight, np.ndarray):
                _size += self.weight.size*4/math.pow(2, 20)
            else:
                _size += self.weight.nnz*4/math.pow(2, 20)
        else:
            raise AssertionError("Classifier is not yet trained!")
        if self.bias is not None:
            if isinstance(self.bias, np.ndarray):
                _size += self.bias.size*4/math.pow(2, 20)
            else:
                _size += self.bias.nnz*4/math.pow(2, 20)
        return _size

    def state_dict(self):
        return dict(
            zip(self.__attrs_to_save,
                itemgetter(*self.__attrs_to_save)(self.__dict__)))

    @property
    def __attrs_to_save(self):
        return ['num_labels', 'valid_labels', 'num_labels_', 'weight', 'bias']

    def save(self, fname):
        pickle.dump(self.state_dict(),
                    open(fname, 'wb'))

    def load(self, fname):
        try:
            for key, val in pickle.load(open(fname, 'rb')).items():
                if key in self.__attrs_to_save:
                    self.__setattr__(key, val)
        except FileNotFoundError:
            print("Model not found at specified path.")

    def __repr__(self):
        return "num_labels: {}, feature_type: {}".format(
            self.num_labels, self.feature_type)

    def evaluate(self, true_labels, predicted_labels):
        # TODO
        raise NotImplementedError("Evaluate yet to be added.")
