#!/bin/bash

timestamp=$(date +%Y%m%d-%H%M%S)
maxepochs=200
split="PDB-RNA"

# reset default configuration and set epochs
sed "s/200/$maxepochs/g" config_default.json > data/config.json
cp mod_redense_default_from_pip_install.pt data/mod_redense.pt

# run training
/home/dmilone/miniconda3/envs/redfold/bin/python ./redfold.py -train \
  ./data/$split/train/ 2>&1 | tee "log/train_${split}_${timestamp}.log"

# set the model to load for testing
ep00=$(printf "%03d" $maxepochs)
sed "s/mod_redense/mod_redense_$ep00/g" config_default.json > data/config.json

# run testing
/home/dmilone/miniconda3/envs/redfold/bin/python ./redfold.py -test \
  ./data/$split/test/ 2>&1 | tee log/test_${split}_${timestamp}.log
