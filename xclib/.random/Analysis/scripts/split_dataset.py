import numpy as np
from xctools.data import data_utils
import glob
import sys
import os
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
data_dir = sys.argv[1]
dset = sys.argv[2]
# print(data_dir)


def write_data(filename, features, labels):
    '''
        Write data in sparse format
        Args:
            filename: str: output file name
            features: csr_matrix: features matrix
            labels: csr_matix: labels matrix

    '''
    with open(filename, 'w') as f:
        out = "{} {} {}".format(
            features.shape[0], features.shape[1], labels.shape[1])
        print(out, file=f)
    with open(filename, 'ab') as f:
        dump_svmlight_file(features, labels, f, multilabel=True)

label_name = glob.glob("%s/%s/labels_split_*.txt" % (data_dir, dset))
feature_splits = list(map(lambda x: np.loadtxt(
    x.replace("labels_", "features_"), dtype=np.int32), label_name))
label_splits = list(map(lambda x: np.loadtxt(x, dtype=np.int32), label_name))

total_splits = len(label_splits)
files = ["test", "train"]

for split_idx in range(total_splits):
    split_id = label_name[split_idx].split(".")[0].split("_")[-1]
    os.makedirs("%s/%s_%s" % (data_dir, dset, split_id), exist_ok=True)
    print("%s/%s_%s" % (data_dir, dset, split_id))

for idx, file in enumerate(files):
    features, labels, num_instances, num_features, num_labels = data_utils.read_data(
        "%s/%s/%s.txt" % (data_dir, dset, file))
    labels = data_utils.binarize_labels(labels, num_labels)
    for split_idx in range(total_splits):
        split_id = label_name[split_idx].split(".")[0].split("_")[-1]
        _features = features[:,feature_splits[split_idx]]
        _labels = labels[:, label_splits[split_idx]]
        f_nnz = np.unique(_features.nonzero()[0])
        l_nnz = np.unique(_labels.nonzero()[0])
        nnz = np.intersect1d(f_nnz,l_nnz)
        _features = _features[nnz,:]
        _labels = _labels[nnz,:]
        write_data("%s/%s_%s/%s.txt" %
                   (data_dir, dset, split_id, file), _features, _labels)
        np.savetxt("%s/%s_%s/%s_%s.txt" %
                   (data_dir, dset, split_id, "valid_indices",file),nnz,fmt="%d")
