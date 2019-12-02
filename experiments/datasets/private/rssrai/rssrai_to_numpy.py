import multiprocessing
import os

import numpy as np
from rssrai import Rssrai


def save_numpy(path, sample, index):
    if not os.path.exists(path):
        os.makedirs(path)
    np.savez(os.path.join(path, f"{index}".zfill(5)), **sample)


def get_valid_numpy(basic_dir, pixel):
    rssrai = Rssrai(mode='val',
                    base_size=None,
                    crop_size=pixel,
                    basic_dir=basic_dir,
                    is_load_numpy=False)
    rssrai_num = len(rssrai)
    numpy_path = os.path.join(basic_dir,
                              f"valid_numpy_{pixel}")
    print(f"rssrai_num {rssrai_num}")
    pool = multiprocessing.Pool(processes=10)
    for epoch in range(1):
        pool.apply_async(epoch_save_numpy, (epoch, numpy_path, rssrai, rssrai_num))
    pool.close()
    pool.join()


def get_train_numpy(basic_dir, pixel):
    rssrai = Rssrai(mode='train',
                    base_size=None,
                    crop_size=pixel,
                    basic_dir=basic_dir,
                    is_load_numpy=False)
    rssrai_num = len(rssrai)
    numpy_path = os.path.join(basic_dir,
                              f"train_numpy_{pixel}")
    print(f"rssrai_num {rssrai_num}")
    pool = multiprocessing.Pool(processes=10)
    for epoch in range(40):
        pool.apply_async(epoch_save_numpy, (epoch, numpy_path, rssrai, rssrai_num))
    pool.close()
    pool.join()


def epoch_save_numpy(epoch, numpy_path, rssrai, rssrai_num):
    for index, sample in enumerate(rssrai):
        print(f"epoch {epoch} {index}/{rssrai_num}")
        sample = rssrai.__getitem__(index, is_save=True)
        save_numpy(numpy_path, sample, epoch * rssrai_num + index)


basic_dir = '/home/deamov/dataset/rssrai2019'

get_valid_numpy(basic_dir, 256)
# get_valid_numpy(512)
# get_train_numpy(basic_dir, 256)
# get_train_numpy(512)
