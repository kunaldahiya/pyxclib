import argparse
import json


__author__ = 'X'


class ParametersBase():
    """
        Base class for parameters in XML
    """
    def __init__(self, description):
        self.parser = argparse.ArgumentParser(description)
        self.params = None

    def _construct(self):
        self.parser.add_argument(
            '--dataset',
            dest='dataset',
            action='store',
            type=str,
            help='dataset name')
        self.parser.add_argument(
            '--data_dir',
            dest='data_dir',
            action='store',
            type=str,
            help='path to main data directory')
        self.parser.add_argument(
            '--model_dir',
            dest='model_dir',
            action='store',
            type=str,
            help='directory to store models')
        self.parser.add_argument(
            '--result_dir',
            dest='result_dir',
            action='store',
            type=str,
            help='directory to store results')
        self.parser.add_argument(
            '--model_fname',
            dest='model_fname',
            default='model',
            action='store',
            type=str,
            help='model file name')
        self.parser.add_argument(
            '--pred_fname',
            dest='pred_fname',
            default='predictions.npy',
            action='store',
            type=str,
            help='prediction file name')
        self.parser.add_argument(
            '--tr_feat_fname',
            dest='tr_feat_fname',
            default='trn_X_Xf.txt',
            action='store',
            type=str,
            help='training feature file name')
        self.parser.add_argument(
            '--val_feat_fname',
            dest='val_feat_fname',
            default='tst_X_Xf.txt',
            action='store',
            type=str,
            help='validation feature file name')
        self.parser.add_argument(
            '--ts_feat_fname',
            dest='ts_feat_fname',
            default='tst_X_Xf.txt',
            action='store',
            type=str,
            help='test feature file name')
        self.parser.add_argument(
            '--tr_label_fname',
            dest='tr_label_fname',
            default='trn_X_Y.txt',
            action='store',
            type=str,
            help='training label file name')
        self.parser.add_argument(
            '--val_label_fname',
            dest='val_label_fname',
            default='tst_X_Y.txt',
            action='store',
            type=str,
            help='validation label file name')
        self.parser.add_argument(
            '--feature_type',
            dest='feature_type',
            default='sparse',
            action='store',
            type=str,
            help='feature type sequential/dense/sparse')
        self.parser.add_argument(
            '--ts_label_fname',
            dest='ts_label_fname',
            default='tst_X_Y.txt',
            action='store',
            type=str,
            help='test label file name')

    def parse_args(self):
        self.params = self.parser.parse_args()

    def load(self, fname):
        vars(self.params).update(json.load(open(fname)))

    def save(self, fname):
        print(vars(self.params))
        json.dump(vars(self.params), open(fname, 'w'), indent=4)
