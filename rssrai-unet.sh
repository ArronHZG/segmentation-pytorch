python   train.py  \
        --model UNet \
        --dataset rssrai \
        --gpu-ids 0 \
        --batch-size=50  \
        --val-batch-size=25 \
        --crop-size=256 \
        --check-point-id=5