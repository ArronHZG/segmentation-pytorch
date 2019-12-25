# cython: language_level=3
import os
import random

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from PIL import Image


def read_csv(path):
    with open(path, "r+") as read:
        lines = read.readlines()
    name_list = [x.split('\n')[0] for x in lines]
    return name_list


class Cloud(data.Dataset):
    NUM_CLASSES = 2

    def __init__(self, mode, base_size, crop_size, basic_dir):

        assert mode in ['train', 'val']
        super().__init__()
        self._base_dir = basic_dir
        self.mode = mode
        self.in_c = 4
        self.mean = (0.52891074, 0.38070734, 0.40119018, 0.36884733)
        self.std = (0.24007008, 0.23784, 0.22267079, 0.21865861)
        self.crop_size = crop_size
        self.base_size = base_size
        self.im_ids = []
        self.images = []
        self.categories = []

        # 加载数据
        if self.mode is 'train':
            csv_path = os.path.join(self._base_dir, "statistic", "train_set.csv")
            self.name_list = pd.read_csv(csv_path)["name"]
            self._image_dir = os.path.join(self._base_dir, 'image')
            self._label_dir = os.path.join(self._base_dir, 'label')
            self.len = len(self.name_list)

        if self.mode is 'val':
            csv_path = os.path.join(self._base_dir, "statistic", "valid_set.csv")
            self.name_list = pd.read_csv(csv_path)["name"]
            self._image_dir = os.path.join(self._base_dir, 'image')
            self._label_dir = os.path.join(self._base_dir, 'label')
            self.len = len(self.name_list)

    def __getitem__(self, index, is_save=False):

        sample = self.transform(self.get_numpy_image(index), is_save=is_save)
        return sample

    def __len__(self):
        return self.len

    def __str__(self):
        return f"[Xian {self.mode}] num_classes:{self.NUM_CLASSES} len: {self.__len__()}"

    def get_numpy_image(self, index):
        '''
        验证集按顺序选取
        测试集按顺序选取
        '''
        sample = None
        if self.mode == 'train':
            sample = self._read_file(self.name_list[index])
            sample = self._random_crop_and_enhance(sample)
        if self.mode == 'val':
            sample = self._read_file(self.name_list[index])
            sample = self._valid_enhance(sample)
        return sample

    def _random_crop_and_enhance(self, sample):
        offset = int(self.crop_size / 10)
        compose = A.Compose([
            A.PadIfNeeded(self.crop_size + offset, self.crop_size + offset),
            A.RandomSizedCrop((self.crop_size - offset + 10, self.crop_size + offset - 10),
                              self.crop_size,
                              self.crop_size, p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RGBShift(),
            A.Blur(),
            A.GaussNoise(),
            A.Normalize(mean=self.mean, std=self.std, p=1)
        ], additional_targets={'image': 'image', 'label': 'mask'})
        return compose(**sample)

    def _valid_enhance(self, sample):
        compose = A.Compose([
            A.CenterCrop(self.crop_size, self.crop_size, p=1),
            A.Normalize(mean=self.mean, std=self.std, p=1)
        ], additional_targets={'image': 'image', 'label': 'mask'})
        return compose(**sample)

    # @functools.lru_cache( maxsize=None )
    def _read_file(self, name):
        image_pil = Image.open(os.path.join(self._image_dir, name))
        image_np = np.array(image_pil)

        label_pil = Image.open(os.path.join(self._label_dir, name))
        label_np = np.array(label_pil)

        return {'image': image_np, 'label': label_np}

    def _get_random_file_name(self):
        return random.choice(self.name_list)

    def transform(self, sample, is_save):
        if not is_save:
            sample['image'] = torch.from_numpy(sample['image']).permute(2, 0, 1)
            sample['label'] = torch.from_numpy(sample['label']).long()
        return sample
