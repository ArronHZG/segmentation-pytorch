python  ../main.py  \
        --model deeplabv3plus \
        --basic-dir /mnt/anna-fuse/taihe/fexion/yaogan/ruiyan_debug/train_data_level_1_1234/split-352 \
        --dataset xian \
        --gpu-ids 0 \
        --batch-size 40 \
        --epochs 1000 \
        --lr 0.01 \
        --crop-size 352