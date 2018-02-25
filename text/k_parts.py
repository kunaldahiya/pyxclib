"""
    Create k parts from given vectorized text and tf-idf features
"""
import sys
import os
sys.path.append('../tools/data')
import data_utils
import pickle
from scipy.sparse import lil_matrix
import numpy as np
import math

def split_train_test(features, labels, split):
    """
        Split k parts in train & test
    """
    k = len(features)
    train_idx = np.where(split == 0)[0].tolist()
    test_idx = np.where(split == 1)[0].tolist()
    train_features = []
    test_features = []
    for i in range(k):
        train_features.append(features[i][train_idx, :])
        test_features.append(features[i][test_idx, :])
    train_labels, test_labels = labels[train_idx, :], labels[test_idx, :]
    return train_features, train_labels, test_features, test_labels

def split_list(a, n):
    """
        Split list a in n parts
    """
    k, m = divmod(len(a), n)
    return list((a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
            for i in range(n)))


def split_doc(vectorized_text, k, features_tfidf):
    features = []
    num_samples, num_features = features_tfidf.shape
    for i in range(k):
        features.append(lil_matrix((num_samples, num_features), dtype=np.float32))

    short_docs = 0
    zero_feat_docs = 0
    for doc_idx in range(num_samples):
        doc_tfidf_feat = features_tfidf[doc_idx, :].nonzero()[1]
        num_features = len(doc_tfidf_feat)
        doc_feat = vectorized_text[doc_idx].copy()
        if num_features < k:
            print("Short document encountered: ", doc_idx)
            short_docs += 1
            try:
                factor = math.ceil(k/num_features)
            except ZeroDivisionError:
                factor = k
                doc_feat = [0]
                features_tfidf[doc_idx, 0] = 1.0
                print("Zero features: ", doc_idx)
                zero_feat_docs += 1
            doc_feat = doc_feat * factor
        splitted_doc_feat = split_list(doc_feat, k)
        for i in range(k):
            keys = splitted_doc_feat[i]
            vals = features_tfidf[doc_idx, keys].toarray()[0, :].tolist()
            features[i][doc_idx, keys] = vals
    for i in range(k):
        features[i] = features[i].tocsr()
    print("#Docs with zero features: {}, #Docs with features less than {}: {}".format(zero_feat_docs, k, short_docs))
    return features


def main():
    data_dir = sys.argv[1]
    vectorized_text = pickle.load(
        open(os.path.join(data_dir, sys.argv[2]), 'rb'))
    tfidf_features = pickle.load(
        open(os.path.join(data_dir, sys.argv[3]), 'rb'))
    k = int(sys.argv[4])
    labels = data_utils.read_sparse_file(os.path.join(data_dir, 'labels.txt'))
    split = data_utils.read_split_file(os.path.join(data_dir, 'split.0.txt'))
    print("Data loaded successfully!")
    features = split_doc(vectorized_text, k, tfidf_features.tocsr())
    train_features, train_labels, test_features, test_labels = split_train_test(features, labels, split)
    with open(os.path.join(data_dir, str(k) + "_parts_train.pkl"), 'wb') as fp:
        pickle.dump({'features': train_features, 'labels': train_labels}, fp)
    with open(os.path.join(data_dir, str(k) + "_parts_test.pkl"), 'wb') as fp:
        pickle.dump({'features': test_features, 'labels': test_labels}, fp)


if __name__ == '__main__':
    main()