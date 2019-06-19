"""
    Generate vectorized text and tf-idf for given text
"""
import os
import sys
import numpy as np
from xctools.text import text_utils
from xctools.data import data_utils
import pickle

def read_split_file(split_fname):
    '''
        Return array of train/ test split
        Args:
            split_fname: str: split file with 0/1 in each line
        Returns:
            np.array: split
    '''
    return np.genfromtxt(split_fname, dtype=np.int)

def _split_data(data, indices):
    """
        Handles list and sparse/dense matrices
        Args:
            data: list of matrix
            indices: indices to choose
        Returns:
            chosen data in the same format
    """
    return list(map(lambda x: data[x], indices))


def split_train_test(features, split):
    '''
        Split a list of text in train and text set
        0: train, 1: test
        Args:
            features: list or dense or sparse matrix
            labels: list or dense or sparse matrix
            split: numpy array with 0/1 split
        Returns:
            train_feat, train_labels: train features and labels
            test_feat, test_labels: test features and labels
    '''
    train_idx = np.where(split == 0)[0].tolist()
    test_idx = np.where(split == 1)[0].tolist()
    train_feat, test_feat = _split_data(features, train_idx), _split_data(
        features, test_idx)
    return train_feat, test_feat

def main():
    data_dir = sys.argv[1]
    data = sys.argv[2]
    # text feature object
    t_obj = text_utils.TextUtility(max_df=0.8, min_df=3)
    #feat_obj =
    t_obj.fit(data)
    keys  = list(t_obj.vocabulary.keys())
    np.savetxt(data_dir+'/Xf.txt',keys,fmt="%s")
    text_utils.save_vocabulary(os.path.join(data_dir, 'vocabulary.json'), t_obj.vocabulary)
    t_obj.save(os.path.join(data_dir, 'text_model.pkl'))
    # Vectorized text as per vocabulary
    vectorized_text = t_obj.transform(data)
    # Save vectorized text
    train,test = split_train_test(vectorized_text,read_split_file(sys.argv[3]))
    tr_lb = data_utils.read_sparse_file(sys.argv[4])
    ts_lb = data_utils.read_sparse_file(sys.argv[5])
    print(tr_lb.shape,len(train))
    print(ts_lb.shape,len(test))
    data_utils.write_data(sys.argv[6],train,tr_lb)
    data_utils.write_data(sys.argv[7],test,ts_lb)

if __name__ == '__main__':
    main()
