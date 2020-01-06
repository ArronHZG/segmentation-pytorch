# SGD
python  ../main_test.py  \
        --model fcn \
        --basic-dir /mnt/anna-fuse/taihe/fexion/yaogan/ruiyan_debug/测试数据 \
        --dataset xian \
        --gpu-ids 0 \
        --batch-size 10 \
        --crop-size 1024 \
        --test-num-classes 5 \
        --lr 0.01 \
        --epochs 1 \
        --check-point-id 12