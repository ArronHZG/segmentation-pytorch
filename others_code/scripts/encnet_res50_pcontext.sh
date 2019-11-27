#!/usr/bin/env bash

#train
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset pcontext \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet50 --checkname encnet_res50_pcontext

#test [single-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset pcontext \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet50 --resume {MODEL} --split val --mode testval

#test [multi-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset pcontext \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet50 --resume {MODEL} --split val --mode testval --ms

#predict [single-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset pcontext \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet50 --resume {MODEL} --split val --mode test

#predict [multi-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset pcontext \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet50 --resume {MODEL} --split val --mode test --ms

#fps
CUDA_VISIBLE_DEVICES=0 python test_fps_params.py --dataset pcontext \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet50
