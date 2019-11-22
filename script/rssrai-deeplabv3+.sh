python  ../train.py  \
        --model deeplabv3plus \
        --dataset rssrai \
        --gpu-ids 0 \
        --batch-size=80 \
        --crop-size=256