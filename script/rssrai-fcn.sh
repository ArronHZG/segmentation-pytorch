python  ../train.py  \
        --model fcn \
        --basic-dir /home/arron/dataset/rssrai2019 \
        --dataset rssrai \
        --gpu-ids 1 \
        --batch-size 100 \
        --crop-size 256 \
        --check-point-id 1
