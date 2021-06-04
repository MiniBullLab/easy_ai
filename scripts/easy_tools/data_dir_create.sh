base_dir="/home/${USER}"
data_dir="$base_dir/easy_data"
if [ ! -d $data_dir ]; then
  mkdir $data_dir
fi
