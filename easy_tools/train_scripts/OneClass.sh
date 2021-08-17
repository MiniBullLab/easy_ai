#!/bin/bash
root_path=$(pwd)

function run_onnx_convert() {
    set -v
    modelDir="./.easy_log/snapshot"
    imageDir="./.easy_log/one_class_img"
    outDir="${root_path}/.easy_log/out"
    modelName=OneClassNet
    outNetName=OneClassNet

    inputColorFormat=1
    outputShape=1,3,224,224
    outputLayerName="o:one_class_output|ot:0,1,2,3|odf:fp32"
    inputDataFormat=0,0,0,0

    mean=123.675,116.28,103.53
    scale=57.63

    rm -rf $outDir
    mkdir $outDir
    mkdir $outDir/dra_image_bin

    #amba
    source /usr/local/amba-cv-tools-2.1.7-20190815.ubuntu-18.04/env/cv22.env

    #cuda10
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

    #caffe
    export PYTHONPATH=/opt/caffe/python:$PYTHONPATH

    ls $imageDir/*.* > $imageDir/img_list.txt
    imgtobin.py -i $imageDir/img_list.txt \
                -o $outDir/dra_image_bin \
                -c $inputColorFormat \
                -d 0,0,0,0 \
                -s $outputShape

    ls $outDir/dra_image_bin/*.bin > $outDir/dra_image_bin/dra_bin_list.txt

    onnxparser.py -m $modelDir/${modelName}.onnx \
                    -i $outDir/dra_image_bin/dra_bin_list.txt \
                    -o $outNetName \
                    -of $outDir/out_parser \
                    -is $outputShape \
                    -im $mean -ic $scale \
                    -iq -idf $inputDataFormat \
                    -odst $outputLayerName

    cd $outDir/out_parser;vas -auto -show-progress $outNetName.vas

    rm -rf ${outDir}/cavalry
    mkdir -p ${outDir}/cavalry

    cavalry_gen -d $outDir/out_parser/vas_output/ \
                -f $outDir/cavalry/$outNetName.bin \
                -p $outDir/ \
                -v > $outDir/cavalry/cavalry_info.txt

    rm -rf vas_output

    cp $outDir/cavalry/$outNetName.bin  ${root_path}/${outNetName}.bin
}

function main() {
    if [ -n "$1" ]; then
        dataset_train_path=$1
    else
        dataset_train_path=/easy_ai/ImageSets/train.txt
    fi

    if [ -n "$2" ]; then
        dataset_val_path=$2
    else
        dataset_val_path=/easy_ai/ImageSets/val.txt
    fi
    echo ${dataset_train_path}
    echo ${dataset_val_path}

    rm -rf ./.easy_log/one_class*
    CUDA_VISIBLE_DEVICES=0 python3 -m easy_tools.easy_ai --task OneClass --gpu 0 --trainPath ${dataset_train_path} --valPath ${dataset_val_path}
    if [ $? -ne 0 ]; then
          echo "Failed to start easy_ai"
          exit -1
    fi
    run_onnx_convert
}

main "$1" "$2"
