# xclib
Tools for multi-label classification problems.

```bash
git clone https://github.com/kunaldahiya/xclib.git
cd xclib
python3 setup.py install --user
```
Usage 
```python
from xclib.data import data_utils

# Read file with features and labels
features, tabels, num_samples, num_features, num_labels = data_utils.read_data('train.txt')

# Read sparse file
labels = data_utils.read_sparse_file('trn_X_Xf.txt', force_header=True)

# Write sparse file
data_utils.write_sparse_file(labels, "labels.txt")
```
