#!/bin/bash
root_path=$(pwd)

function run_caffe_convert() {
    set -v
    modelDir="./.easy_log/snapshot"
    imageDir="./.easy_log/det_img"
    outDir="${root_path}/.easy_log/out"
    caffeNetName=denet
    outNetName=denet

    inputColorFormat=0
    outputShape=1,3,416,416
    outputLayerName0="o:det_output0|odf:fp32"
    outputLayerName1="o:det_output1|odf:fp32"
    outputLayerName2="o:det_output2|odf:fp32"
    inputDataFormat=0,0,8,0

    mean=0.0
    scale=255.0

    rm -rf $outDir
    mkdir $outDir
    mkdir $outDir/dra_image_bin

    #amba
    source /usr/local/amba-cv-tools-2.2.1-20200928.ubuntu-18.04/env/cv22.env

    #cuda10
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

    export LD_LIBRARY_PATH=/usr/local/cyberRT/third_party/gflags/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/cyberRT/third_party/glog/lib:$LD_LIBRARY_PATH

    #caffe
    export PYTHONPATH=/opt/caffe/python:$PYTHONPATH

    ls $imageDir/*.* > $imageDir/img_list.txt

    imgtobin.py -i $imageDir/img_list.txt \
                -o $outDir/dra_image_bin \
                -c $inputColorFormat \
                -d 0,0,0,0 \
                -s $outputShape

    ls $outDir/dra_image_bin/*.bin > $outDir/dra_image_bin/dra_bin_list.txt

    caffeparser.py -p $modelDir/$caffeNetName.prototxt \
                   -m $modelDir/$caffeNetName.caffemodel \
                   -i $outDir/dra_image_bin/dra_bin_list.txt \
                   -o $outNetName \
                   -of $outDir/out_parser \
                   -it 0,1,2,3 \
                   -iq -idf $inputDataFormat -odst $outputLayerName0 -odst $outputLayerName1 -odst $outputLayerName2 # -c act-force-fx16,coeff-force-fx16

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

function run_onnx_convert() {
    set -v
    modelDir="./.easy_log/snapshot"
    imageDir="./.easy_log/det_img"
    outDir="${root_path}/.easy_log/out"
    caffeNetName=denet
    outNetName=denet

    inputColorFormat=0
    outputShape=1,3,416,416
    outputLayerName0="o:det_output0|odf:fp32"
    outputLayerName1="o:det_output1|odf:fp32"
    outputLayerName2="o:det_output2|odf:fp32"
    inputDataFormat=0,0,8,0

    mean=0.0
    scale=255.0

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
                    -it 0,1,2,3 \
                    -iq -idf $inputDataFormat -odst $outputLayerName0 -odst $outputLayerName1 -odst $outputLayerName2
                    # -c act-force-fx16,coeff-force-fx16


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
        dataset_train_path=/easy_ai/ImageSets/train.txt
    fi

    if [ -n "$2" ]; then
        dataset_val_path=$2
    else
        dataset_val_path=/easy_ai/ImageSets/val.txt
    fi
    echo ${dataset_train_path}
    echo ${dataset_val_path}

    rm -rf ./.easy_log/detect2d*

    CUDA_VISIBLE_DEVICES=0 python3 -m easy_tools.easy_ai --task DeNet --gpu 0 --trainPath ${dataset_train_path} --valPath ${dataset_val_path}
    if [ $? -ne 0 ]; then
          echo "Failed to start easy_ai"
          exit -1
    fi
    python3 -m easy_tools.easy_convert --task DeNet --input ./.easy_log/snapshot/denet.onnx
    if [ $? -ne 0 ]; then
          echo "Failed to start easy_convert"
          exit -2
    fi
    run_caffe_convert
}

main "$1" "$2"
