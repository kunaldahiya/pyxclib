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
TOKENIZER=$(pwd)/scripts/py/raw_tf_idf_Y.py

work_dir="${HOME}/scratch/lab/Workspace"
datasets="${work_dir}/data/${1}"

ROOT=$datasets
mkdir -p $ROOT
X=$datasets'/corpus_labels_text.txt'
XT=$datasets'/corpus_labels_titles.txt'
TLOWX=$ROOT'/low_labels_X.txt'
TEMP=$ROOT'/temp.txt'
mkdir -p $ROOT'/Yfts'
clean_text(){
	echo "USING STD TOKENIZER"
	python -u $TOKENIZER $1 $2 $3 $4 $5 $6 $7
}

# awk -F '->' '{$1=""; print $0}' $X | perl $PREPROCESS >$TLOWX'Text'
# awk -F '->' '{$1=""; print $0}' $XT | perl $PREPROCESS >$TLOWX'Title'
# paste -d' ' $TLOWX'Title' $TLOWX'Text' > $TLOWX
# cat $X | perl $PREPROCESS >$TLOWX
clean_text $ROOT'/Yfts' $TLOWX $ROOT'/Yfts/Yf.txt'
# rm -rf $TLOWX
# rm -rf $TLOWX'Text'
# rm -rf $TLOWX'Title'