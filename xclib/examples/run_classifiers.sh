# * ./run_classifiers.sh <dataset> <method> <feature_type> <A> <B>
#     - dataset: dataset name (see structure)
#     - method: ova/slice
#     - feature_type: sparse/dense 
#         (change 'trn_ft_file' and 'tst_ft_file' accordingly)
#     - A & B: for propensity scored metrics

# * Example: ./run_classifiers.sh EURLex-4K ova sparse 0.55 1.5

# Change 'work_dir', 'trn_ft_file', and 'trn_ft_file' (if required)

# * Structure: 
# work_dir
#    - programs
#    - models
#    - data
#      - EURLex-4K
#    - results

# * Files in data directory (e.g. work_dir/data/EURLex-4K) (for default case):
#     - train features and labels: trn_X_Xf.txt, trn_X_Y.txt
#     - test features and labels: tst_X_Xf.txt, tst_X_Y.txt

# * If train.txt and test.txt are available (use following commands to split)
# perl train.txt trn_X_Xf.txt trn_X_Y.txt
# perl test.txt tst_X_Xf.txt tst_X_Y.txt


train (){
    # $1 : dataset
    # $2 : data_dir
    # $3 : model_dir
    # $4 : clf_type
    # $5 : train feat file name
    # $6 : train label file name
    # $7 : optional params
    python3 -W ignore run_classifiers.py -mode 'train' \
    --dataset $1 \
    -clf_type $4 \
    --data_dir $2 \
    --model_dir $3 \
    --tr_feat_fname $5 \
    --tr_label_fname $6 \
    --num_threads 25 \
    --batch_size 500 \
    $7
}

predict () {
    # $1: dataset
    # $2: data_dir
    # $3: model_dir
    # $4: result_dir
    # $5: clf_type
    # $6: feature file name
    # $7: label file name
    # $8: opt_params
    python3 -W ignore run_classifiers.py -mode 'predict' \
    --dataset $1 \
    --data_dir "${2}" \
    --model_dir "${3}" \
    -clf_type $5 \
    --ts_feat_fname $6 \
    --ts_label_fname $7 \
    --result_dir "${4}" \
    --num_threads 12 \
    --batch_size 1000 \
    $8
}

evaluate () {
    # $1 train
    # $2 target
    # $3 prediction
    # $4 A
    # $5 B
    python3 evaluate.py $1 $2 $3 $4 $5
}

dataset=$1
work_dir="/home/XC"
A=$4
B=$5
method=$2
feature_type=$3

data_dir="${work_dir}/data/${dataset}"

model_dir="${work_dir}/models/${method}/${dataset}"
result_dir="${work_dir}/results/${method}/${dataset}"
score_file="${result_dir}/score.txt"
log_train="${result_dir}/log_train.txt"
log_predict="${result_dir}/log_predict.txt"

mkdir -p "${model_dir}"
mkdir -p "${result_dir}"

# Adjust these file-names as per available data
trn_ft_file="${data_dir}/trn_X_Xf.txt"
trn_lbl_file="${data_dir}/trn_X_Y.txt"
tst_ft_file="${data_dir}/tst_X_Xf.txt"
tst_lbl_file="${data_dir}/tst_X_Y.txt"

feat_params="--norm l2 --feature_type ${feature_type}"
loss_params="--threshold 0.01 --max_iter 50 --dual True"
short_params="--M 100 --efC 300 --efS 300 --num_neighbours 300 --feature_type ${feature_type}"
params="${feat_params} ${loss_params} ${opt_params}"

train $dataset "${data_dir}" "${model_dir}" $method "${trn_ft_file}" "${trn_lbl_file}" "${params}" | tee -a "${log_train}"
predict $dataset "${data_dir}" "${model_dir}" "${result_dir}" $method $tst_ft_file $tst_lbl_file "${params}" | tee -a "${log_predict}"
evaluate ${trn_lbl_file} ${tst_lbl_file} ${score_file} $A $B
