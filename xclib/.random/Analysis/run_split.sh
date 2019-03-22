
wrk_dir="${HOME}/scratch/lab/Workspace"
data_dir="${wrk_dir}/data"
dset=$1

python scripts/split_dataset.py "${data_dir}" "${dset}"
