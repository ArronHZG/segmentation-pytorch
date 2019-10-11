#python   train.py  \
#        --model Nested \
#        --dataset rssrai \
#        --gpu-ids 0 \
#        --batch-size=70  \
##        --val-batch-size=25 \
#        --crop-size=256 \
##        --check-point-id=5
#
python -m torch.distributed.launch --nproc_per_node=2 train.py \
        --model UNet_Nested \
        --dataset rssrai \
        --gpu-ids 0,1 \
        --batch-size=30 \
        --crop-size=256 \
        --check-point-id=1