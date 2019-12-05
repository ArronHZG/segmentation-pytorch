python  ../train.py  \
        --model resunet \
        --basic-dir /mnt/anna-fuse/taihe/fexion/yaogan/ruiyan_debug/train_data/split-352 \
        --dataset xian \
        --gpu-ids 0 \
        --batch-size 3 \
        --epochs 1000 \
        --lr 0.01 \
        --crop-size 352