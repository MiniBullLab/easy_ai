base_dir="/home/${USER}"
data_dir="${base_dir}/easy_data"
classnet_dir="${data_dir}/classnet"
denet_dir="${data_dir}/denet"
segnet_dir="${data_dir}/segnet"

if [ ! -d $data_dir ]; then
  mkdir $data_dir
fi

if [ ! -d $classnet_dir ]; then
  mkdir $classnet_dir
  if [ ! -d "${classnet_dir}/JPEGImages" ]; then
    mkdir "${classnet_dir}/JPEGImages"
  if
fi

if [ ! -d $denet_dir ]; then
  mkdir $denet_dir
  if [ ! -d "${denet_dir}/JPEGImages" ]; then
    mkdir "${denet_dir}/JPEGImages"
  if
  if [ ! -d "${denet_dir}/Annotations" ]; then
    mkdir "${denet_dir}/Annotations"
  if
fi

if [ ! -d $segnet_dir ]; then
  mkdir $segnet_dir
  if [ ! -d "${segnet_dir}/JPEGImages" ]; then
    mkdir "${segnet_dir}/JPEGImages"
  if
  if [ ! -d "${segnet_dir}/SegmentLabel" ]; then
    mkdir "${segnet_dir}/SegmentLabel"
  if
fi
