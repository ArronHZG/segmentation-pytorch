python  ../train.py  \
        --model deeplabv3plus \
        --basic-dir /home/deamov/dataset/rssrai2019 \
        --dataset rssrai \
        --gpu-ids 1 \
        --batch-size 80 \
        --crop-size 256