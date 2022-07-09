#!/bin/bash
root_path=$(pwd)

function run_onnx_convert() {
    set -v
    modelDir="./.easy_log/snapshot"
    imageDir="./.easy_log/ocr_det_img"
    outDir="${root_path}/.easy_log/out"
    modelName=OCRDeNet
    outNetName=OCRDeNet

    inputColorFormat=1
    outputShape=1,3,480,480
    outputLayerName="o:ocr_denet_output|ot:0,1,2,3|odf:fp32"
    inputDataFormat=0,0,0,0

    mean=0.0
    scale=255.0

    rm -rf $outDir
    mkdir -m 755 $outDir
    rm -rf $outDir/dra_image_bin
    mkdir -m 755 -p $outDir/dra_image_bin

    #amba
    source /usr/local/amba-cv-tools-2.2.1-20200928.ubuntu-18.04/env/cv22.env

    #cuda10
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

    #caffe
    export LD_LIBRARY_PATH=/opt/caffe/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/opt/caffe/python:$PYTHONPATH

    graph_surgery.py onnx -m $modelDir/${modelName}.onnx  -t Default
    mv $modelDir/${modelName}.onnx $modelDir/${modelName}_raw.onnx
    mv $modelDir/${modelName}_modified.onnx $modelDir/${modelName}.onnx

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
                    -odst $outputLayerName \
                    -c act-allow-fp16,coeff-force-fx16

    cd $outDir/out_parser;vas -auto -show-progress $outNetName.vas

    rm -rf ${outDir}/cavalry
    mkdir -p ${outDir}/cavalry

    cavalry_gen -d $outDir/out_parser/vas_output/ \
                -f $outDir/cavalry/$outNetName.bin \
                -p $outDir/ \
                -V 2.1.7 \
                -v > $outDir/cavalry/cavalry_info.txt

    rm -rf vas_output

    cp $outDir/cavalry/$outNetName.bin  ${root_path}/${outNetName}.bin
}

function main() {
    if [ -n "$1" ]; then
        dataset_train_path=$1
    else
        dataset_train_path=/easy_data/ImageSets/train.txt
    fi

    if [ -n "$2" ]; then
        dataset_val_path=$2
    else
        dataset_val_path=/easy_data/ImageSets/val.txt
    fi
    echo ${dataset_train_path}
    echo ${dataset_val_path}

    rm -rf ./.easy_log/polygon2d*
    CUDA_VISIBLE_DEVICES=0 python3 -m easy_tools.easy_ai --task OCRDenet --gpu 0 --trainPath ${dataset_train_path} --valPath ${dataset_val_path}

    if [ $? -ne 0 ]; then
          echo "Failed to start easy_ai"
          exit -1
    fi
    run_onnx_convert
}

main "$1" "$2"