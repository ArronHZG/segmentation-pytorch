import os
from glob import glob

import numpy as np
import torch

from experiments.datasets.path import Path

crop_size = 256
base_dir = Path.db_root_dir('rssrai')


def load_numpy(name_list, index):
    i = index
    d = None
    while d is None:
        try:
            sample = np.load(name_list[i])
            d = {'image': torch.from_numpy(sample['image']), "label": torch.from_numpy(sample['label']).long()}
        except:
            print(f"{name_list[i]} is bad! auto remove it.")
            # os.remove(path_list[i])
            # path_list[i] = path_list[0]
    return d


path_list = glob(os.path.join(base_dir, f'test_train_numpy_{crop_size}', '*.npz'))
print(path_list)
print(len(path_list))
for i in range(len(path_list)):
    d = load_numpy(path_list, i)
    print(d['image'].shape)
