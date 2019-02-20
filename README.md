# xctools
Tools for multi-label classification problems.

```bash
git clone https://github.com/kunaldahiya/xctools.git
cd xctools
python setup.py install --user
```
Usage
```python
from xctools.data import data_utils as du.
sparse_file = du.read_sparse_file('lbl.txt',header=True)
ft,lb,num_samples,num_features,num_labels = du.read_data('train.txt')
lb = du.binarize_labels(lb,num_labels)
```
