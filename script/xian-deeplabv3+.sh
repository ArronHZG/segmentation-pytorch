python  ../train.py  \
        --model deeplabv3plus \
        --basic-dir /mnt/anna-fuse/taihe/fexion/yaogan/ruiyan_debug/train_data/split-352 \
        --dataset xian \
        --gpu-ids 0 \
        --batch-size 20 \
        --epochs 1000 \
        --lr 0.01 \
        --crop-size 352