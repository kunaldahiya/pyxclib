#!/usr/bin/env bash

: '

Created by XC group

$1 -> 	Dataset
$2  -> 	1, to build bag of words features else give 0
$3 ->	1, remove empty lines (containing 0 labels or 0 words)
$4 ->   1, when data for each doc is confined within <text> </text> else 0

complete_doc.txt ->	contains single file of raw data for both test and train in format 
					(id->text) with not headers
split.0.txt      -> contains split of test and train with 0 for train
labels.txt 		 -> contains single file if raw data having ids of label present in each 
					document with headers present (<Total docs> <Total labels>)	


OUTPUTS
Generates following files in $1"data/"
Train
	1) train.txt -- contains encoded data for RNN
	2) train_build.txt -- contains encoded data for Bag of words
	3) trn_lbl_mat.txt -- contains raw data where each line in the document is delimited by " . "
	4) trn_ft.txt -- contains labels data with headers (<Total docs> <Total labels>)

Test
	1) test.txt -- contains encoded data for RNN
	2) test_build.txt -- contains encoded data for Bag of words
	3) tst_lbl_mat.txt -- contains raw data where each paragraph is delimited by " . "
	4) tst_ft.txt -- contains labels data with headers (<Total docs> <Total labels>)
'

if [ $4 -eq 0 ]
then
	PREPROCESS=$(pwd)/scripts/pl/elseProcess.perl
	#statements
else
	PREPROCESS=$(pwd)/scripts/pl/wikiProcess.perl
fi

TOKENIZER=$(pwd)/scripts/py/nltk_tokenize.py
BUILDVOCAB=$(pwd)/scripts/py/build_vocab.py
FEATURES=$(pwd)/scripts/py/build_features.py
BUILDSTD=$(pwd)/scripts/py/build_std_data.py

work_dir="${HOME}/scratch/lab/Workspace"

datasets="${work_dir}/data/${1}/"
ROOT=$datasets'data/'
mkdir -p $ROOT
X=$datasets'complete_doc.txt'
SPLIT=$datasets'split.0.txt'
LABELS=$datasets'labels.txt'
TLOWX=$ROOT'low_X.txt'
TEMPX=$ROOT'temp_X.txt'
TEMPLB=$ROOT'temp_lb.txt'
TRAINFT=$ROOT'trn_ft.txt'
TRAINLB=$ROOT'trn_lbl_mat.txt'
TESTFT=$ROOT'tst_ft.txt'
TESTLB=$ROOT'tst_lbl_mat.txt'
VOCAB=$ROOT'tf_idf.txt'
WORDS=$ROOT'Xf.txt'

get_features(){
	head -n 1 $1 | awk -F ' ' '{print $2}'
}

get_lines(){
	wc -l $1 | awk -F ' ' '{print $1}'
}

build_labels(){
	echo -e $(get_lines $1) $(get_features $LABELS)>temp.txt
	cat $1 >> temp.txt
	mv temp.txt $1
}

clean_text(){
	echo "USING NLTK TOKENIZER"
	python -u $TOKENIZER $1 $2 $3 $4
	# cp $1 $2
}

echo "Creating Complete corpus"
cat $datasets'/train_map.txt' $datasets'/test_map.txt' >$X
echo "CLEANING DATA"
awk -F '->' '{print $2}' $X | perl $PREPROCESS >$TLOWX
clean_text $TLOWX $TEMPX

awk -F '->' '{print $2}' $datasets'train_map.txt' | perl $PREPROCESS > $datasets'temp.txt'
clean_text $datasets'temp.txt' $TRAINFT
rm -rf $datasets'temp.txt'

awk -F '->' '{print $2}' $datasets'test_map.txt' | perl $PREPROCESS > $datasets'temp.txt'
clean_text $datasets'temp.txt' $TESTFT
rm -rf $datasets'temp.txt'

tail -n +2 $datasets'train.txt'| awk -F ' ' '{print $1}'> $TRAINLB
tail -n +2 $datasets'test.txt'| awk -F ' ' '{print $1}'> $TESTLB

echo $(get_lines $TRAINLB)
echo -e $(get_lines $TRAINLB) $(head -n 1 $datasets'train.txt'| awk -F ' ' '{print $3}')>temp.txt
cat $TRAINLB >> temp.txt
mv temp.txt $TRAINLB

echo $(get_lines $TESTLB)
echo -e $(get_lines $TESTLB) $(head -n 1 $datasets'test.txt'| awk -F ' ' '{print $3}')>temp.txt
cat $TESTLB >> temp.txt
mv temp.txt $TESTLB

echo "CREATING DATASET"
python $BUILDVOCAB $ROOT '3' $VOCAB
python $FEATURES $TRAINFT $TRAINLB $ROOT'train_X.txt' $VOCAB $3
python $FEATURES $TESTFT $TESTLB $ROOT'test_X.txt' $VOCAB $3

echo $(expr $(wc -l $ROOT'train_X.txt'|awk -F ' ' '{print $1}') - 1 ) $(head -n 1 $ROOT'train_X.txt') > $ROOT'train.txt'
tail -n +2 $ROOT'train_X.txt' >> $ROOT'train.txt'

echo $(expr $(wc -l $ROOT'test_X.txt'|awk -F ' ' '{print $1}') - 1 ) $(head -n 1 $ROOT'test_X.txt') > $ROOT'test.txt'
tail -n +2 $ROOT'test_X.txt' >> $ROOT'test.txt'


awk -F ' ' '{print $1}' <$VOCAB > $WORDS

head -n 1 $ROOT'train.txt'
head -n 1 $ROOT'test.txt'

rm -rf $TEMPX
rm -rf $TEMPLB
rm -rf $ROOT'train_X.txt'
rm -rf $ROOT'test_X.txt'
rm -rf $TLOWX
rm -rf $TRAINFT
rm -rf $TESTFT
rm -rf $TRAINLB
rm -rf $TESTLB

if [ $2 -eq 1 ]
then
	echo "Building Standard Datasets"
	python $BUILDSTD $ROOT'train.txt' $ROOT'train_build.txt' $3
	python $BUILDSTD $ROOT'test.txt' $ROOT'test_build.txt' $3
	head -n 1 $ROOT'train_build.txt'
	head -n 1 $ROOT'test_build.txt'

fi 