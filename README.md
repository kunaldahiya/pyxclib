# pyxclib

Tools for extreme multi-label classification problems.

```bash
git clone https://github.com/kunaldahiya/pyxclib.git
cd pyxclib
python3 setup.py install --user
```

Usage

## Data reading/writing

```python
from xclib.data import data_utils

# Read file with features and labels (old format from XMLRepo)
features, tabels, num_samples, num_features, num_labels = data_utils.read_data('train.txt')

# Read sparse file (see docstring for more)
# header can be set to false (if required)
labels = data_utils.read_sparse_file('trn_X_Xf.txt', header=True)

# Write sparse file (with header)
data_utils.write_sparse_file(labels, "labels.txt")
```

## Evaluation

Implementation of precision, nDCG, propensity scored precision/nDCG and recall is included

```python
from xclib.data import data_utils
import xclib.evaluation.xc_metrics as xc_metrics

# Read ground truth and predictions
true_labels = data_utils.read_sparse_file('tst_X_Y.txt')
predicted_labels = data_utils.read_sparse_file('parabel_predictions.txt')

# evaluate (See examples/evaluate.py for more details)
acc = xc_metrics.Metrics(true_labels=true_labels)
args = acc.eval(predicted_labels, 5)
print(xc_metrics.format(*args))

```

## Tools

* _sparse_/_dense_: topk, rank, binarize, sigmoid, normalize, _etc_.
* _dense_: topk, binarize, sigmoid, normalize, _etc_.
* _shortlist_: Shortlist, ShortlistCentroids, ShortlistInstances, _etc_.
* _analysis_: compare_predictions, compare_nearest_neighbors, _etc_.
