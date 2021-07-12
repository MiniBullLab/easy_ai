#!/bin/bash
set -v
root_path=$(pwd)
modelDir="./.easy_log/snapshot"
imageDir="./.easy_log/text_img"
outDir="${root_path}/.easy_log/out"
tensorflowName=TextNet
outNetName=TextNet

inputColorFormat=1
outputShape=1,3,32,128
inputLayerName="i:input.1|is:1,3,32,128"
compare_inputLayerName="i:net_input=${imageDir}/img_list.txt|iq|idf:0,0,0,0|is:1,3,32,128"
temp_outputLayerName="682"
outputLayerName="o:682|ot:0,1,2,3|odf:fp32"
inputDataFormat=0,0,0,0

mean=127.5,127.5,127.5
scale=127.5

rm -rf $outDir
mkdir $outDir
mkdir $outDir/dra_image_bin

#amba
source /usr/local/amba-cv-tools-2.2.1-20200928.ubuntu-18.04/env/cv22.env

#cuda10
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#caffe
export LD_LIBRARY_PATH=/opt/caffe/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/caffe/python:$PYTHONPATH

ls $imageDir/*.* > $imageDir/img_list.txt
imgtobin.py -i $imageDir/img_list.txt \
            -o $outDir/dra_image_bin \
            -c $inputColorFormat \
            -d 0,0,0,0 \
            -s $outputShape

ls $outDir/dra_image_bin/*.bin > $outDir/dra_image_bin/dra_bin_list.txt
graph_surgery.py tf -p $modelDir/$tensorflowName.pb \
                    -o $modelDir/${tensorflowName}_temp.pb \
                    -isrc $inputLayerName \
                    -on $temp_outputLayerName \
                    -t ConstantifyShapes

CUDA_VISIBLE_DEVICES=-1
tfparser.py -p $modelDir/${tensorflowName}_temp.pb \
               -i $outDir/dra_image_bin/dra_bin_list.txt \
               -o $outNetName \
               -of $outDir/out_parser \
               -is $outputShape \
               -im $mean -ic $scale \
               -iq -idf $inputDataFormat \
               -odst $outputLayerName \
               -c act-allow-fp16,coeff-force-fx16

cd $outDir/out_parser;vas -auto -show-progress $outNetName.vas

rm -rf ${outDir}/cavalry
mkdir -p ${outDir}/cavalry

cavalry_gen -d $outDir/out_parser/vas_output/ \
            -f $outDir/cavalry/$outNetName.bin \
            -p $outDir/ \
            -v > $outDir/cavalry/cavalry_info.txt

rm -rf vas_output
