python  ../train.py \
        --model fcn \
        --basic-dir /home/deamov/dataset/VOC2012/VOC2012 \
        --dataset pascal_voc \
        --gpu-ids 0,1 \
        --apex 1 \
        --epochs 500 \
        --batch-size 32 \
        --base-size 520 \
        --crop-size 512