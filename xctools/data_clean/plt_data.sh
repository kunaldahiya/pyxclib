#!/usr/bin/env bash
DATABUILD=$(pwd)'/scripts/py/plt_data.py'
dataset_dir=$1
data_out_dir=$2
dataset_name=$3
raw='_raw'
mkdir -p "${data_out_dir}/${dataset_name}/"

head -n 1 $dataset_dir'/train$raw.txt' |awk -F ' ' '{ print $1 }' >"${data_out_dir}/${dataset_name}/${dataset_name}_train.examples"
head -n 1 $dataset_dir'/train$raw.txt' |awk -F ' ' '{ print $2 }' >"${data_out_dir}/${dataset_name}/${dataset_name}_train.features"
head -n 1 $dataset_dir'/train$raw.txt' |awk -F ' ' '{ print $3 }' >"${data_out_dir}/${dataset_name}/${dataset_name}_train.labels"
python $DATABUILD $1'/train$raw.txt' $data_out_dir $dataset_name 'train'


head -n 1 $dataset_dir'/test$raw.txt' |awk -F ' ' '{ print $1 }' >"${data_out_dir}/${dataset_name}/${dataset_name}_test.examples"
head -n 1 $dataset_dir'/test$raw.txt' |awk -F ' ' '{ print $2 }' >"${data_out_dir}/${dataset_name}/${dataset_name}_test.features"
head -n 1 $dataset_dir'/test$raw.txt' |awk -F ' ' '{ print $3 }' >"${data_out_dir}/${dataset_name}/${dataset_name}_test.labels"
python $DATABUILD $1'/test$raw.txt' $data_out_dir $dataset_name 'test'


# head -n 1 $dataset_dir'/test_raw.txt' |awk -F ' ' '{print $1 >'"${data_out_dir}/${dataset_name}/${dataset_name}_test.examples"'; print $2 >'"${data_out_dir}/${dataset_name}/${dataset_name}_test.features"'; print $3 >'"${data_out_dir}/${dataset_name}/${dataset_name}_test.labels"'; }'
# python $DATABUILD $dataset_dir'/test_raw.txt' $data_out_dir $dataset_name 'test'
