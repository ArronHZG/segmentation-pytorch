python  train.py  \
        --model FCN \
        --dataset rssrai \
        --gpu-ids 0 \
        --batch-size=50  \
        --crop-size=256 \
        --epochs 300 \
        --check-point-id=1