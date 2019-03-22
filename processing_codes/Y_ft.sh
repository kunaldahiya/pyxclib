#!/usr/bin/env bash

: '

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
TOKENIZER=$(pwd)/scripts/py/tf_idf_Y.py

work_dir="${HOME}/scratch/lab/Workspace"
datasets="${work_dir}/data/${1}"

ROOT=$datasets'/Yfts'
mkdir -p $ROOT
X=$datasets'/Y.txt'
TLOWX=$ROOT'/low_X.txt'
Y=$ROOT'/Yf.txt'
VocabY=$ROOT'/VocabY.txt'
clean_text(){
	echo "USING STD TOKENIZER"
	python -u $TOKENIZER $1 $2 $3 $4
}


perl $PREPROCESS $X > $TLOWX
# cat $X > $TLOWX
clean_text $ROOT $TLOWX $Y $VocabY
# rm -rf $TLOWX