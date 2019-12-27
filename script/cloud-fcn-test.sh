# SGD
python  ../main-test.py  \
        --model fcn \
        --basic-dir /mnt/anna-fuse/taihe/fexion/yaogan/ruiyan_debug/source_data \
        --dataset cloud \
        --gpu-ids 0 \
        --batch-size 10 \
        --crop-size 1024 \
        --test-num-classes 2 \
        --check-point-id 4