from xclib.data import data_utils as du
from scipy.sparse import csr_matrix
import numpy as np
import re
import _pickle as p
import os

Y = open("labels.txt", "r").readlines()
data = p.load(open("corpus.pkl", "rb"))


# abstract = open("abstract.txt","w")
# title = open("title.txt","w")
# text = open("text.txt","w")
# labels = open("labels.txt","w")
# for (key,val) in data.items():
#     if len(val["SA"])>0:
#         lbls = np.unique(val["SA"])
#         _doc = val["text"].replace("\n","")
#         _id = val["id"]
#         _title = key
#         _labels = '_^_'.join(lbls)
#         _abstract = _doc.split('==')[0]
#         print("%s->%s"%(_id, _title), file=title, flush=True)
#         print("%s->%s"%(_id, _doc), file=text, flush=True)
#         print("%s->%s"%(_id, _abstract), file=abstract, flush=True)
#         print("%s->%s"%(_id, _labels), file=labels, flush=True)

labels_text = {}
dict_labels_freq = {}

for y in Y:
    lbs = np.unique(('->'.join(y.split('->')[1:])).strip().split('_^_'))
    for lb in lbs:
        dict_labels_freq[lb] = dict_labels_freq.get(lb, 0)+1

for key in dict_labels_freq.keys():
        labels_text[key] = data.get(key, {"text": ""})[
                                    'text'].strip().replace("\n", "").split("==")[0]

def _build_dataset(keys, name="Y", title="title.txt", text="text.txt", labels="labels.txt"):
        os.makedirs(name, exist_ok=True)
        print("Total keys are %d"%len(keys))
        Y_dict = {}
        Y_not_text = open("%s/labels_text_not_found.txt" % (name), "w")
        Y_lbs = open("%s/label_text_found.txt" % (name), "w")
        Yf = open("%s/label_text.txt" % (name), "w")
        not_feature = []
        k_in = 0
        for idx, lbs in enumerate(keys):
                _text = labels_text.get(lbs, "")
                if _text == "":
                        not_feature.append(lbs)
                else:
                        Y_dict[lbs] = k_in
                        print("%d->%s" % (k_in, lbs), file=Y_lbs, flush=True)
                        print("%d->%s . %s" % (k_in, lbs, _text), file=Yf, flush=True)
                        k_in += 1

        for lbs in not_feature:
                Y_dict[lbs] = k_in
                print("%d->%s" % (k_in, lbs), file=Y_not_text, flush=True)
                k_in += 1
        total_sentences = 0
        label_col = []
        label_row = []
        label_data = []
        corpus_text = open("%s/corpus_text.txt" % (name), "w")
        corpus_title = open("%s/corpus_title.txt" % (name), "w")
        with open(title, "r") as tit, open(text, "r") as txt, open(labels, "r") as lbs:
                for _title in tit:
                        _title = ('->'.join(_title.split('->')[1:])).strip()
                        _text = (
                            '->'.join(txt.readline().split('->')[1:])).strip()
                        _label = np.unique(
                            ('->'.join(lbs.readline().split('->')[1:])).strip().split('_^_'))
                        valid_label = list(
                            [Y_dict[lb] for lb in _label if Y_dict.get(lb, -1) > -1])
                        if len(valid_label) > 0:
                                label_col.extend(valid_label)
                                label_row.extend(
                                    [total_sentences]*len(valid_label))
                                label_data.extend(
                                    [np.int32(1)]*len(valid_label))
                                print("%s" %
                                      (_title), file=corpus_title, flush=True)
                                print("%s . %s" %
                                      (_title, _text), file=corpus_text, flush=True)
                                total_sentences += 1
        mat = csr_matrix((label_data, (label_row, label_col)), dtype=np.int32)
        du.write_sparse_file(mat, "%s/corpus_lbl_mat.txt" % (name))

_build_dataset([key for key in dict_labels_freq.keys() if dict_labels_freq[key] > 1], name="Y")
_build_dataset([key for key in dict_labels_freq.keys() if dict_labels_freq[key] == 1], name="Y-zero")
