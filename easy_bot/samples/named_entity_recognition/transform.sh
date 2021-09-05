#!/bin/bash

python3 transform.py

echo "======================================================="
echo "aws s3 cp --recursive train.txt s3://your-bucket/samples/bosonnlp/"
echo "aws s3 cp --recursive val.txt s3://your-bucket/samples/bosonnlp/"
echo "======================================================="
