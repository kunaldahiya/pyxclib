#!/usr/bin//env bash

:/ '

Created by XC group

$1 -> 	Dataset

complete_doc.txt ->	contains single file of raw data for both test and train in format 
					(id->text) with not headers
split.0.txt      -> contains split of test and train with 0 for train
labels.txt 		 -> contains single file if raw data having ids of label present in each 
					document with headers present (<Total docs> <Total labels>)	

'

PREPROCESS=$(pwd)/scripts/pl/elseProcess.perl
SEP_LBL_FTS=$(pwd)/scripts/pl/convert_format.pl
TOKENIZER=$(pwd)/scripts/py/run_vec_tfidf.py

work_dir="${HOME}/scratch/lab/Workspace"
datasets="${work_dir}/data/${1}"

ROOT=$datasets'/data'
mkdir -p $ROOT
X=$datasets'/complete_doc.txt'
SPLIT=$datasets'/split.0.txt'
TLOWX=$ROOT'/low_X.txt'
TRAINLB=$ROOT'/trn_lbl_mat.txt'
TRAINFT=$ROOT'/trn_fts_mat.txt'
TESTLB=$ROOT'/tst_lbl_mat.txt'
TESTFT=$ROOT'/tst_fts_mat.txt'

TEMP=$ROOT'/temp.txt'

clean_text(){
	echo "USING STD TOKENIZER"
	python -u $TOKENIZER $1 $2 $3 $4 $5 $6 $7
}

perl $SEP_LBL_FTS "${datasets}/train.txt" \
                "${TRAINFT}" \
                "${TRAINLB}"


perl $SEP_LBL_FTS "${datasets}/test.txt" \
                "${TESTFT}" \
                "${TESTLB}"
exit
rm -rf "${TEMP}"

awk -F '->' '{print $2}' $X | perl $PREPROCESS >$TLOWX
clean_text $ROOT $TLOWX $SPLIT "${TRAINLB}" "${TESTLB}" $ROOT'/train.txt' $ROOT'/test.txt'
