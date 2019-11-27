import multiprocessing
import os

import numpy as np
from torch.utils.data import DataLoader

from experiments.datasets import Rssrai
from experiments.datasets.path import Path


def save_numpy(path, sample, index):
    if not os.path.exists(path):
        os.makedirs(path)
    np.savez(os.path.join(path, f"{index}".zfill(5)), **sample)


def get_valid_numpy(pixel):
    rssrai = Rssrai(mode='val', base_size=None, crop_size=pixel)
    rssrai_num = len(rssrai)
    numpy_path = os.path.join(Path.db_root_dir('rssrai'),
                              f"valid_numpy_{pixel}")
    print(f"rssrai_num {rssrai_num}")

    for index, sample in enumerate(rssrai):
        print(f"{index}/{rssrai_num}")
        sample = rssrai.__getitem__(index, is_save=True)
        save_numpy(numpy_path, sample, index)


def get_train_numpy(pixel):
    rssrai = Rssrai(mode='train', base_size=None, crop_size=pixel)
    rssrai_num = len(rssrai)
    numpy_path = os.path.join(Path.db_root_dir('rssrai'),
                              f"train_numpy_{pixel}")
    print(f"rssrai_num {rssrai_num}")

    train_loader = DataLoader(rssrai,
                              batch_size=30,
                              pin_memory=True,
                              num_workers=8)

    pool = multiprocessing.Pool(processes=10)
    for epoch in range(300):
        pool.apply_async(epoch_save_numpy, (epoch, numpy_path, rssrai, rssrai_num))
    pool.close()
    pool.join()


def epoch_save_numpy(epoch, numpy_path, rssrai, rssrai_num):
    for index, sample in enumerate(rssrai):
        print(f"epoch {epoch} {index}/{rssrai_num}")
        sample = rssrai.__getitem__(index, is_save=True)
        save_numpy(numpy_path, sample, epoch * rssrai_num + index)


# get_valid_numpy(256)
# get_valid_numpy(512)
# get_train_numpy(256)
get_train_numpy(512)
