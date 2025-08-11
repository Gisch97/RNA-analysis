#!/bin/bash

timestamp=$(date +%Y%m%d-%H%M%S)
maxepochs=200
split="5_hl_folds"
fold=0
hls=(5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95)

for hl in "${hls[@]}"
do
    # reset default configuration and set epochs
    sed "s/200/$maxepochs/g" config_default.json > data/config.json
    cp mod_redense_default_from_pip_install.pt data/mod_redense.pt

    #echo ./data/$split/hl$hl/$fold/train/

    # run training
    export CUDA_VISIBLE_DEVICES=0; \
    /home/dmilone/miniconda3/envs/redfold/bin/python ./redfold.py -train \
      ./data/$split/hl$hl/$fold/train/ 2>&1 | tee "log/train_${split}_hl${hl}_${fold}_${timestamp}.log"

    # set the model to load for testing
    ep00=$(printf "%03d" $maxepochs)
    sed "s/mod_redense/mod_redense_$ep00/g" config_default.json > data/config.json

    # run testing
    export CUDA_VISIBLE_DEVICES=0; \
    /home/dmilone/miniconda3/envs/redfold/bin/python ./redfold.py -test \
      ./data/$split/hl$hl/$fold/test/ 2>&1 | tee log/test_${split}_hl${hl}_${fold}_${timestamp}.log
done