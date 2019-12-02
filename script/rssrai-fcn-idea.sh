python  ../train.py \
        --model fcn-idea \
        --basic-dir /home/deamov/dataset/rssrai2019 \
        --dataset rssrai \
        --gpu-ids 2 \
        --apex 2 \
        --epochs 1000 \
        --batch-size 100 \
        --crop-size 256
