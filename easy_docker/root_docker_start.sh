#!/usr/bin/env bash
easy_path=/home/${USER}/easy_data
if [ ! -d $easy_path ];then
   mkdir $easy_path
else
   echo $easy_path
fi

docker run -it --shm-size="2g" --gpus=all -v ${easy_path}:/easy_data easy_ai
