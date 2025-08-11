#!/bin/bash

timestamp="20250109-210448"
maxepochs=200
split="bprna"

# reset default configuration and set epochs
#sed "s/200/$maxepochs/g" config_default.json > data/config.json
#cp mod_redense_default_from_pip_install.pt data/mod_redense.pt

# run training
#/home/dmilone/miniconda3/envs/redfold/bin/python ./redfold.py -train \
#  ./data/$split/TR0_VL0/ 2>&1 | tee "log/train_${split}_${timestamp}.log"

# set the model to load for testing
sed "s/mod_redense/mod_redense_200_bpRNA_TR0VL0_250110/g" config_default.json > data/config.json

# run testing
/home/dmilone/miniconda3/envs/redfold/bin/python ./redfold.py -test \
  ./data/$split/TS0/ 2>&1 | tee log/test_${split}_TS0_${timestamp}.log
