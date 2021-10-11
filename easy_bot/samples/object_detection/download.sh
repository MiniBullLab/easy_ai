#!/bin/bash

if [ ! -f tiny_motorbike.zip ]; then
    wget https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip
fi
rm -rf tiny_motorbike
unzip tiny_motorbike.zip
