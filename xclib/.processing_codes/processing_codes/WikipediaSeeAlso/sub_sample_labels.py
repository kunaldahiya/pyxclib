import xclib.data.data_utils as du
import numpy as np
import sys
import os

def _sample_labels(labels_corpus, initial_labels=100):
    lbl = du.read_sparse_file(labels_corpus, force_header=True).tolil()
    feq_lbl = np.ravel(lbl.sum(axis=0))**0.75
    p_lbl = feq_lbl/np.sum(feq_lbl)
    lbl_idx = np.arange(lbl.shape[1])
    sample_lb = np.random.choice(
        lbl_idx, size=initial_labels, p=p_lbl, replace=False)
    print(sample_lb.size)
    return sample_lb


def _create_sample(sample_lb, data_dir, prefix, data_folder):
    files = ['test.txt', 'train.txt']
    for file in files:
        _fts, _lbs, _, _, _ = du.read_data(os.path.join(data_dir, data_folder, file))
        _lbs = _lbs[:, sample_lb]
        valid_instances = np.where(np.ravel(_lbs.sum(axis=1)) > 0)[0]
        _fts = _fts[valid_instances,:]
        _lbs = _lbs[valid_instances,:]
        os.makedirs(os.path.join(data_dir, prefix, data_folder), exist_ok=True)
        du.write_data(os.path.join(data_dir, prefix, data_folder, file), features=_fts, labels=_lbs)
    vocab_X = np.loadtxt(os.path.join(data_dir, data_folder, 'Xf.txt'), dtype=str)
    np.savetxt(os.path.join(data_dir, prefix, data_folder, 'Xf.txt'), vocab_X, fmt="%s")
    
def _create_sample_Y(sample_lb, data_dir, prefix, data_folder):
    Yf = du.read_sparse_file(os.path.join(data_dir, data_folder, 'Yf.txt'))
    VocabY = np.loadtxt(os.path.join(data_dir, data_folder, 'vocabY.txt'), dtype=str)
    Yf = Yf[sample_lb,:]
    valid_words = np.where(np.ravel(Yf.sum(axis=0))>0)[0]
    Yf = Yf[:, valid_words]
    VocabY = VocabY[valid_words]
    os.makedirs(os.path.join(data_dir, prefix, data_folder), exist_ok=True)
    du.write_sparse_file(Yf, os.path.join(data_dir, prefix, data_folder, 'Yf.txt'))
    np.savetxt(os.path.join(data_dir, prefix, data_folder, 'vocabY.txt'), VocabY, fmt="%s")



if __name__ == '__main__':
    data_dir = sys.argv[1]
    labels_corpus = sys.argv[2]
    initial_labels = int(sys.argv[3])
    prefix = sys.argv[4]
    data_text = sys.argv[5]
    data_title = sys.argv[6]
    Y = sys.argv[7]
    Y_titles = sys.argv[8]
    sample_lb = _sample_labels(os.path.join(data_dir, labels_corpus), initial_labels)
    _create_sample(sample_lb, data_dir, prefix, data_title)
    _create_sample(sample_lb, data_dir, prefix, data_text)
    _create_sample_Y(sample_lb, data_dir, prefix, Y)
    _create_sample_Y(sample_lb, data_dir, prefix, Y_titles)

