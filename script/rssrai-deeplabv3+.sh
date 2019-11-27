python  ../train.py  \
        --model deeplabv3plus \
        --dataset rssrai \
        --gpu-ids 2 \
        --batch-size 100 \
        --crop-size 256