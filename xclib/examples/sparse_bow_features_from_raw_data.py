"""
    Generate tf-idf features for given text
"""
import sys
from xclib.data import data_utils
from xclib.utils.text import BoWFeatures
from xclib.utils.sparse import ll_to_sparse


def read(fname):
    labels = []
    text = []
    fp = data_utils.read_corpus(fname)
    for line in fp:
        # different dataset might use different fields
        text.append(line['title'])
        labels.append(line['target_ind'])
    return text, labels


def max_feature_index(trn_labels, tst_labels):
    max_ind = max([max(item) for item in trn_labels])
    return max(max_ind, max([max(item) for item in tst_labels]))


def process(trn_fname, tst_fname, encoding='latin',
            min_df=2, dtype='float32'):
    trn_text, trn_labels = read(trn_fname)

    # feature extractor
    fex = BoWFeatures(encoding=encoding, min_df=min_df, dtype=dtype)
    fex.fit(trn_text)

    # get features and labels for train set
    trn_features = fex.transform(trn_text)
    del trn_text

    # do test
    tst_text, tst_labels = read(tst_fname)
    tst_features = fex.transform(tst_text)
    del tst_text

    # Ensures both have same number of labels
    max_ind = max_feature_index(trn_labels, tst_labels)
    trn_labels = ll_to_sparse(
        trn_labels, shape=(len(trn_labels), max_ind))
    tst_labels = ll_to_sparse(
        tst_labels, shape=(len(tst_labels), max_ind))
    return trn_features, trn_labels, tst_features, tst_labels


def main():
    trn_ifname = sys.argv[1]
    tst_ifname = sys.argv[2]
    trn_ofname = sys.argv[3]
    tst_ofname = sys.argv[4]
    # Read data and create features
    trn_features, trn_labels, tst_features, tst_labels = process(
        trn_ifname, tst_ifname)

    # write the data
    data_utils.write_data(trn_ofname, trn_features, trn_labels)
    data_utils.write_data(tst_ofname, tst_features, tst_labels)


if __name__ == "__main__":
    main()
