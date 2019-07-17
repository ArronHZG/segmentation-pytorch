from __future__ import print_function, division

import glob
import os
import pandas as pd
import numpy as np
import scipy.io
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
from dataloaders import custom_transforms as tr
import albumentations as A
from mypath import Path
from tqdm import tqdm


class RssraiTestSet(data.Dataset):
    NUM_CLASSES = 16

    def __init__(self,
                 base_dir=Path.db_root_dir('rssrai')
                 ):
        super().__init__()
        self._base_dir = base_dir
        test_path = os.path.join(self._base_dir, "test_npy")
        test_path_output = os.path.join(self._base_dir, "test_output_npy")
        self._mean_path = os.path.join(self._base_dir, "mean.npy")
        self._std_path = os.path.join(self._base_dir, "std.npy")
        self.mean = None
        self.std = None
        self.images = None
        self.test_size = 0.1

        # 文件夹不存在保护
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        if not os.path.exists(test_path_output):
            os.makedirs(test_path_output)
        # 加载数据
        self.images = glob.glob(os.path.join(test_path, '*'))
        # 得到均值和方差
        self._get_mean_std()
        # Display stats
        print('Number of images: {:d}'.format(len(self.images)))

    def _get_mean_std(self):
        try:
            print("loading mean and std value")
            self.mean = np.load(self._mean_path)
            self.std = np.load(self._std_path)
        except:
            raise ("need to compute mean and std")
        print(f"mean: {self.mean}")
        print(f"std: {self.std}")

    def __getitem__(self, index):
        _img = self._read_numpy_file(index)
        sample={"name":self.images[index],"image": self.transform(_img)}
        return sample

    def __len__(self):
        return len(self.images)

    def _read_numpy_file(self, index):
        return np.load(self.images[index]).astype('float64') / 255

    def transform(self, test_image):
        image = A.Normalize(mean=self.mean, std=self.std)(image=test_image)["image"]
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        return image

    def __repr__(self):
        return 'Rssrai(split=test)'


if __name__ == '__main__':
    from dataloaders import custom_transforms as tr
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    rssrai_val = RssraiTestSet()
    for sample in rssrai_val:
        print(sample.size())
        break
