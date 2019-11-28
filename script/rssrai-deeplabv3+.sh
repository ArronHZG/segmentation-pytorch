python  ../train.py  \
        --model deeplabv3plus \
        --basic-dir /home/arron/dataset/rssrai2019 \
        --dataset rssrai \
        --gpu-ids 2 \
        --batch-size 100 \
        --crop-size 256