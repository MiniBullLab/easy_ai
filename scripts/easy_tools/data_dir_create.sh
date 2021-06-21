base_dir="/home/${USER}"
data_dir="${base_dir}/easy_data"
classnet_dir="${data_dir}/classnet"
denet_dir="${data_dir}/denet"
segnet_dir="${data_dir}/segnet"
if [ ! -d $data_dir ]; then
  mkdir $data_dir
fi
