#!/bin/bash
root_path=$(pwd)

function run_onnx_convert() {
    set -v
    modelDir="./.easy_log/snapshot"
    imageDir="./.easy_log/text_img"
    outDir="${root_path}/.easy_log/textnet_out"
    modelName=TextNet
    outNetName=TextNet

#    inputColorFormat=1
#    outputShape=1,3,32,128

#    mean=127.5,127.5,127.5
#    scale=127.5
    inputLayerName="i:text_input=${outDir}/dra_image_bin/dra_bin_list.txt|is:1,3,32,128|idf:0,0,0,0|iq|im:127.5,127.5,127.5|ic:127.5"
    outputLayerName="o:text_output|ot:0,1,2,3|odf:fp32"

    rm -rf $outDir
    mkdir -m 755 $outDir
    rm -rf $outDir/dra_image_bin
    mkdir -m 755 -p $outDir/dra_image_bin

    #amba
    source /usr/local/amba-cv-tools-2.2.1-20200928.ubuntu-18.04/env/cv25.env

    #cuda10
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

    #caffe
    export LD_LIBRARY_PATH=/opt/caffe/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/opt/caffe/python:$PYTHONPATH

    graph_surgery.py onnx -m $modelDir/${modelName}.onnx  -t Default
    mv $modelDir/${modelName}.onnx $modelDir/${modelName}_raw.onnx
    mv $modelDir/${modelName}_modified.onnx $modelDir/${modelName}.onnx

#    ls $imageDir/*.* > $imageDir/img_list.txt
#    imgtobin.py -i $imageDir/img_list.txt \
#                -o $outDir/dra_image_bin \
#                -c $inputColorFormat \
#                -d 0,0,0,0 \
#                -s $outputShape
#    ls $outDir/dra_image_bin/*.bin > $outDir/dra_image_bin/dra_bin_list.txt

    gen_image_list.py -f $imageDir \
                      -o $imageDir/img_list.txt \
                      -ns -e jpg -c 0 -d 0,0 -r 32,128 \
                      -bf $outDir/dra_image_bin \
                      -bo $outDir/dra_image_bin/dra_bin_list.txt

    rm -rf ${outDir}/out_parser
    onnxparser.py -m $modelDir/${modelName}.onnx \
                  -o $outNetName \
                  -of ${outDir}/out_parser \
                  -isrc ${inputLayerName} \
                  -odst $outputLayerName \
                  -c act-allow-fp16,coeff-force-fx16 \
                  -dinf cerr

    # Vas Compiler
    rm -rf ${outDir}/out_parser/vas_output
    cd ${outDir}/out_parser
    vas -auto -show-progress ${outNetName}.vas
    cd -

    # Run Ades
    rm -rf ${outDir}/ades_output
    mkdir -m 755 -p ${outDir}/ades_output
    ades_autogen.py -v ${modelName}  \
                    -p ${outDir}/out_parser \
                    -l ${outDir}/ades_output  \
                    -ib text_input=$(cat ${outDir}/dra_image_bin/dra_bin_list.txt | head -1)
    cd ${outDir}/ades_output
    ades ${modelName}_ades.cmd
    cd -

    # Run layer_compare.py
    rm -rf ${outDir}/layer_compare
    mkdir -m 755 -p ${outDir}/layer_compare
    layer_compare.py onnx -m ${modelDir}/${modelName}.onnx \
                          -isrc ${inputLayerName} \
                          -c act-force-fx16,coeff-force-fx16 \
                          -odst ${outputLayerName} \
                          -n ${modelName} \
                          -v ${outDir}/out_parser \
                          -o ${outDir}/layer_compare/layer_compare \
                          -d 3
    mv lc_cnn_output preproc -t ${outDir}/layer_compare/

    rm -rf ${outDir}/cavalry
    mkdir -m 755 -p ${outDir}/cavalry
    cavalry_gen -d $outDir/out_parser/vas_output/ \
                -f $outDir/cavalry/$outNetName.bin \
                -p $outDir/ \
                -V 2.1.7 \
                -v > $outDir/cavalry/cavalry_info.txt
    echo  $(cat ${outDir}/dra_image_bin/dra_bin_list.txt | head -1) | xargs -n 1 echo | xargs -i cp -rf {} ${outDir}/cavalry/

    cp $outDir/cavalry/$outNetName.bin  ${root_path}/${outNetName}.bin
    rm -rf logs lc_cnn_output lc_onnx_output ades
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

    rm -rf ./.easy_log/recognize_text*
    CUDA_VISIBLE_DEVICES=0 python3 -m easy_tools.easy_ai -t TextNet -g 0 -i ${dataset_train_path} -v ${dataset_val_path}
    if [ $? -ne 0 ]; then
        echo "Failed to start easy_ai"
        exit -1
    fi
    run_onnx_convert
}

main "$1" "$2"