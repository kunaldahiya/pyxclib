# xctools
Tools for multi-label classification problems.

```bash
git clone https://github.com/kunaldahiya/xctools.git
cd xctools
python setup.py install --user
```
Usage 
```python
from xctools.data import data_utils as du
lb_sparse_file = du.read_sparse_file('trn_lbl_mat.txt',header=True)
ft_sparse_file = du.read_sparse_file('trn_ft_mat.txt',header=True)
ft,lb,num_samples,num_features,num_labels = du.read_data('train.txt')
lb = du.binarize_labels(lb,num_labels)
```
