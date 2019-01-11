
module load apps/pythonpackages/3.6.0/pytorch/0.4.1/gpu
module load apps/pythonpackages/3.6.0/torchvision/0.2.1/gpu

wrkspace="${HOME}/scratch/lab/Workspace"
dataset=$1
topK=5
format=$2

result_dir="${wrkspace}/results/ANALYSIS/${dataset}"
model_dir="${wrkspace}/models/ANALYSIS/${dataset}"
data_dir="${wrkspace}/data/${dataset}"

mkdir -p $result_dir
mkdir -p $model_dir

test_file=$data_dir'/test.txt'
train_file=$data_dir'/train.txt'
strig=''
color=('-g' '-r' '-y' '-c')
c_indx=0

for file in $(ls $model_dir | egrep '\.mat$')
do
    label=$(echo $file | awk -F '.' '{print $1}')
    string=$string" ${label} "${color[c_indx]}" ${model_dir}/${file}"
    ((c_indx++))
done

params="${test_file} ${train_file} ${topK} ${result_dir}/${dataset}${format} ${string}"
echo $params
python scripts/contribution.py $params
