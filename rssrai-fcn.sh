python -m torch.distributed.launch --nproc_per_node=2 train.py  \
        --model FCN \
        --dataset rssrai \
        --gpu-ids 0,1 \
        --batch-size=60  \
#        --val-batch-size=25 \
        --crop-size=256 \
        --check-point-id=1