python -m torch.distributed.launch --nproc_per_node=2 main_amp.py -a resnet50 --b 224 --workers 4 --opt-level O2 ./
