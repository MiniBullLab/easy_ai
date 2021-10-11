#!/usr/bin/env bash

set -v
set -e

rm -rf data
mkdir -p ./data/shopeeiet/train
cp demo-shopee-iet-competition.zip ./data/shopeeiet/
cd ./data/shopeeiet/
unzip -q demo-shopee-iet-competition.zip
mv Data-Shopee-IET\ ML\ Competition.zip train
cd train && unzip -q Data-Shopee-IET\ ML\ Competition.zip && cd ..
mv train/Test test
rm *.zip *.csv train/*.zip

set +v

echo "======================================================="
echo "cd data && aws s3 cp --recursive shopeeiet s3://your-bucket/samples/shopeeiet"
echo "======================================================="
