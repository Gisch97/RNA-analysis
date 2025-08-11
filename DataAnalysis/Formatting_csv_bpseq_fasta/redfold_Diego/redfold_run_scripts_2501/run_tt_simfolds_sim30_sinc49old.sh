#!/bin/bash

timestamp=$(date +%Y%m%d-%H%M%S)
maxepochs=200
split="6_sim_folds"
sim=30
folds=(0 1 2 3 4)

for i in "${folds[@]}"
do
    # reset default configuration and set epochs
    sed "s/200/$maxepochs/g" config_default.json > data/config.json
    cp mod_redense_default_from_pip_install.pt data/mod_redense.pt

    #ls ./data/$split/$i/train/

    # run training
    /home/dmilone/miniconda3/envs/redfold/bin/python ./redfold.py -train \
      ./data/$split/sim$sim/$i/train/ 2>&1 | tee "log/train_${split}_sim${sim}_${i}_${timestamp}.log"

    # set the model to load for testing
    ep00=$(printf "%03d" $maxepochs)
    sed "s/mod_redense/mod_redense_$ep00/g" config_default.json > data/config.json

    # run testing
    /home/dmilone/miniconda3/envs/redfold/bin/python ./redfold.py -test \
      ./data/$split/sim$sim/$i/test/ 2>&1 | tee log/test_${split}_sim${sim}_${i}_${timestamp}.log
done
