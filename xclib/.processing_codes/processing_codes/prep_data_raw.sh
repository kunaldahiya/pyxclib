#!/usr/bin/sh

wrk_dir="${HOME}/scratch/lab/Workspace"
data_sir="${wrk_dir}/data"
dset=$1

ROOT="${data_dir}/dset"
corpus=$2

python scripts/py/pyPrep.py $corpus $ROOT
python scripts/py/pyLabels.py $ROOT
python scripts/py/pyToSparse.py $ROOT
python scripts/py/dataSplitter.py $ROOT
python scripts/py/create_test_train.py $ROOT
# perl elseProcess.perl < sa.txt 