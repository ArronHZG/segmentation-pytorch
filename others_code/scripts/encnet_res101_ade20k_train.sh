#!/usr/bin/env bash

#train
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset ade20k \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet101 --checkname encnet_res101_ade20k_train

#test [single-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset ade20k \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet101 --resume {MODEL} --split val --mode testval

#test [multi-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset ade20k \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet101 --resume {MODEL} --split val --mode testval --ms

#predict [single-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset ade20k \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet101 --resume {MODEL} --split val --mode test

#predict [multi-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset ade20k \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet101 --resume {MODEL} --split val --mode test --ms
