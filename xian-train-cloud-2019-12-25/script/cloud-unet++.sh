# SGD
python  ../main.py  \
        --model unet-nested \
        --basic-dir /mnt/anna-fuse/taihe/fexion/yaogan/ruiyan_debug/train_data_cloud/split-352 \
        --dataset cloud \
        --gpu-ids 0 \
        --batch-size 20 \
        --epochs 20 \
        --lr 0.01 \
        --crop-size 352

