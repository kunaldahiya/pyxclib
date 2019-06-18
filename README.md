# xclib
Tools for multi-label classification problems.

```bash
git clone https://github.com/kunaldahiya/xclib.git
cd xclib
sh run.sh
```
Usage 
```python
from xclib.data import data_utils
from xclib.classifier.parabel import Parabel
trn_ft, trn_lb, _, _, _ = data_utils.read_data('train.txt')
outfile = "/path/to/model/directory"
clf = Parabel(outfile)
clf.fit(trn_ft, trn_lb)
tst_ft, _, _, _, _ = data_utils.read_data('test.txt')
score_mat = clf.predict(tst_ft)
data_utils.write_sparse_file(score_mat, "out.txt")
```
