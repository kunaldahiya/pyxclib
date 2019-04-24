SEVENZ="${HOME}/scratch/p7zip/bin/7z"
WRK_DIR="${HOME}/scratch/lab/Workspace/data"
data_directory="${WRK_DIR}/stack"
for file in $(ls $data_directory/7zipfiles | egrep '\.7z$')
do
    label=$(echo $file | awk -F '.7z' '{print $1}')
    echo $label $file
    7z x $data_directory/7zipfiles/$file -o$data_directory/data/$label
done
