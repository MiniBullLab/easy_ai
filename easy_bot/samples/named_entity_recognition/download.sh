#!/bin/bash

rm -f BosonNLP_NER_6C.zip
rm -rf BosonNLP_NER_6C
rm -rf __MACOSX
wget https://static.bosonnlp.com/resources/BosonNLP_NER_6C.zip
unzip BosonNLP_NER_6C.zip
mv BosonNLP_NER_6C/BosonNLP_NER_6C.txt origindata.txt
rm -rf __MACOSX

