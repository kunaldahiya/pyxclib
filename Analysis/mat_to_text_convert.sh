data_dir="${1}"

for file in $(ls $data_dir | egrep '\.mat$')
do
    label=$(echo $file | awk -F '.' '{print $1}')
    echo $file
    python3 scripts/mat_to_txt.py "${data_dir}/${file}" "${data_dir}/${label}.txt"

done