# Convert given files into train and test

import sys
import data_utils
import pickle

def main():
    feat_file = sys.argv[1]
    labels_file = sys.argv[2]
    split_file = sys.argv[3]
    with open(feat_file, 'rb') as fp:
        feat = pickle.load(fp).tocsr()
    labels = data_utils.read_sparse_file(labels_file)
    split = data_utils.read_split_file(split_file)

    train_feat, train_labels, test_feat, test_labels = data_utils.split_train_test(feat, labels, split)
    data_utils.write_data(sys.argv[4], train_feat, train_labels)
    data_utils.write_data(sys.argv[5], test_feat, test_labels)


if __name__ == '__main__':
    main()