# xclib
Tools for multi-label classification problems.

```bash
git clone https://github.com/kunaldahiya/xclib.git
cd xclib
sh run.sh
```
Usage 
```python
from xclib.data import data_utils as du
from xclib.classifier.parabel import Parabel
lb_sparse_file = du.read_sparse_file('trn_lbl_mat.txt', header=True)
ft_sparse_file = du.read_sparse_file('trn_ft_mat.txt', header=True)
ft, lb, num_samples, num_features, num_labels = du.read_data('train.txt')
lb = du.binarize_labels(lb,num_labels)
outfile = "/path/to/model/directory"
clf = Parabel(outfile)
clf.fit(ft,lb)
score_mat = clf.predict(ft_sparse_file)
du.write_sparse_file(score_mat, out.txt)
```
