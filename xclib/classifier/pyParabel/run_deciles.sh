#!/bin/bash
# Create baselines for different embeddings
convert () {
    data_tools="/home/cse/phd/anz168048/scratch/Workspace/tools/data"
	perl $data_tools/convert_format.pl $1 $2 $3
	perl $data_tools/convert_format.pl $4 $5 $6

}

train_parabel () {
	./parabel_train $1 $2 $3 -T 3 -s 0 -t $4
}

predict_parabel () {
	./parabel_predict $1 $2 $3
}

run_parabel () {
    # $1 train feature file
    # $2 train label file
    # $3 test feture file
    # $4 test label file
    # $5 num trees
    # $6 model directory
    # $7 score file
    # $8 log train file
    # $9 log predict
    # $10 log evaluate
    echo "Training.."
    train_parabel $1 $2 $6 $5 > $8
    echo "Predicting.."
    predict_parabel $3 $6 $7 >> $9
    echo "Evaluating.."
    evaluate $7 $4 >> ${10}
}

evaluate () {
    eval_tools="/home/cse/phd/anz168048/scratch/Workspace/tools/evaluation"
    #chmod +x $eval_tools/prec_k
    #chmod +x $eval_tools/nDCG_k
    $eval_tools/prec_k $1 $2 5 0
    $eval_tools/nDCG_k $1 $2 5 0
}

compile () {
    make
}

#datasets=("Amazon-670K-deciles")
datasets=("EURLex-4K-deciles")
work_dir="/home/cse/phd/anz168048/scratch/Workspace/xc_sparse"
deciles=(0 1)
dset_idx=0
#compile
for dset in ${datasets[*]}
do
    for dec in ${deciles[*]}
    do  
        echo "Running datatset: ${dset}, Decile: ${dec}"
        data_dir="${work_dir}/data/${dset}"
        model_dir="${work_dir}/models/${dset}/Parabel/split_${dec}"
        results_dir="${work_dir}/results/${dset}/Parabel/split_${dec}"
        mkdir -p $model_dir
        mkdir -p $results_dir
	
        train_file="${data_dir}/train_split_${dec}.txt"
        test_file="$data_dir/test_split_${dec}.txt"
        trn_ft_file="${data_dir}/tr_X_Xf.txt"
        trn_lbl_file="${data_dir}/tr_X_Y.txt"
        tst_ft_file="${data_dir}/ts_X_Xf.txt"
        tst_lbl_file="${data_dir}/ts_X_Y.txt"
        convert ${train_file} ${trn_ft_file} ${trn_lbl_file} ${test_file} ${tst_ft_file} ${tst_lbl_file}


        num_trees=3
        model_dir="${model_dir}"
        score_file="${results_dir}/score.txt"
        log_train="${results_dir}/log_train.txt"
        log_predict="${results_dir}/log_predict.txt"
        log_evaluate="${results_dir}/log_evaluate.txt"
        run_parabel $trn_ft_file $trn_lbl_file $tst_ft_file $tst_lbl_file $num_trees $model_dir $score_file $log_train $log_predict $log_evaluate
        rm $trn_ft_file $tst_ft_file $trn_lb_file $tst_lb_file
        # Delete the models (too much space is required)
        rm $model_dir/*.tree
	done
	((dset_idx = dset_idx + 1))
done
