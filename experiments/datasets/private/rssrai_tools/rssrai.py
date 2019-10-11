import os
import random
from glob import glob

import albumentations as A
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import pandas as pd

from experiments.utils.tools import make_sure_path_exists
from .rssrai_utils import mean, std, encode_segmap
from ...path import Path


class Rssrai(data.Dataset):
    NUM_CLASSES = 16

    def __init__(self, mode='train', base_size=256, crop_size=256, base_dir=Path.db_root_dir('rssrai'),
                 is_load_numpy=False):

        assert mode in ['train', 'val']
        super().__init__()
        self._base_dir = base_dir
        self.mode = mode
        self.in_c = 4
        self.mean = mean
        self.std = std
        self.crop_size = crop_size
        self.val_crop_size = 512
        self.base_size = base_size
        self.im_ids = []
        self.images = []
        self.categories = []
        self.is_load_numpy = is_load_numpy
        self.numpy_path = os.path.join(self._base_dir, f"train_numpy_{self.crop_size}")
        make_sure_path_exists(self.numpy_path)

        # 加载数据
        if self.mode == 'train' and self.is_load_numpy is False:
            train_csv = os.path.join(self._base_dir, 'train_set.csv')
            self._label_name_list = pd.read_csv(train_csv)["文件名"].values.tolist()
            # self._label_path_list = glob(os.path.join(self._base_dir, 'split_train_520', 'label', '*.tif'))
            # self._label_name_list = [name.split('/')[-1] for name in self._label_path_list]
            self._image_dir = os.path.join(self._base_dir, 'split_train', 'img')
            self._label_dir = os.path.join(self._base_dir, 'split_train', 'label')
            self.len = 100000

        if self.mode == 'train' and self.is_load_numpy is True:
            self.path_list = glob(os.path.join(self._base_dir, f'train_numpy_{self.crop_size}', '*.npz'))
            self.len = len(self.path_list)

        if self.mode == 'val':
            self._label_path_list = glob(os.path.join(self._base_dir, 'split_val_256', 'label', '*.tif'))
            self._label_name_list = [name.split('/')[-1] for name in self._label_path_list]
            self._image_dir = os.path.join(self._base_dir, 'split_val_256', 'img')
            self._label_dir = os.path.join(self._base_dir, 'split_val_256', 'label')
            self.len = len(self._label_name_list)

    def __getitem__(self, index):
        if self.is_load_numpy is False:
            sample = self.transform(self.get_numpy_image(index))
            self.save_numpy(sample)
        else:
            sample = self.load_numpy(index)
        return sample

    def __len__(self):
        return self.len

    def __str__(self):
        return f"[Rssrai {self.mode}] num_classes:{self.NUM_CLASSES} len: {self.__len__()}"

    def get_numpy_image(self, index):
        '''
        训练集随机选一张图片,然后随机crop
        验证集按顺序选取
        测试集按顺序选取
        '''
        sample = None
        if self.mode == 'train':
            name = self._get_random_file_name()
            sample = self._read_file(name)
            sample = self._random_crop_and_enhance(sample)
        if self.mode == 'val':
            sample = self._read_file(self._label_name_list[index])
            sample = self._valid_enhance(sample)
        return sample

    def _random_crop_and_enhance(self, sample):
        compose = A.Compose([
            A.PadIfNeeded(self.base_size, self.base_size, p=1),
            A.RandomSizedCrop((self.crop_size - 100, self.crop_size + 100), self.crop_size, self.crop_size, p=1),
            # A.RandomCrop(self.crop_size, self.crop_size, p=1),
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
            # A.PadIfNeeded(self.base_size, self.base_size, p=1),
            # A.CenterCrop(self.val_crop_size, self.val_crop_size, p=1),
            A.Normalize(mean=self.mean, std=self.std, p=1)
        ], additional_targets={'image': 'image', 'label': 'mask'})
        return compose(**sample)

    # @functools.lru_cache( maxsize=None )
    def _read_file(self, label_name):
        image_name = label_name.replace("_label", "")
        image_pil = Image.open(os.path.join(self._image_dir, image_name))
        image_np = np.array(image_pil)

        label_pil = Image.open(os.path.join(self._label_dir, label_name))
        label_np = np.array(label_pil)
        label_mask = encode_segmap(label_np)

        return {'image': image_np, 'label': label_mask}

    def _read_test_file(self, image_name):
        image_pil = Image.open(os.path.join(self._image_dir, image_name))
        image_np = np.array(image_pil)

        return {'image': image_np, 'name': image_name}

    def _get_random_file_name(self):
        return random.choice(self._label_name_list)

    def transform(self, sample):
        sample['image'] = torch.from_numpy(sample['image']).permute(2, 0, 1)
        if self.mode != "test":
            sample['label'] = torch.from_numpy(sample['label']).long()
        return sample

    def save_numpy(self, sample):
        np.savez_compressed(os.path.join(self.numpy_path, str(hash(sample["image"]))), **sample)

    def load_numpy(self, index):
        i = index
        d = None
        while d is None:
            try:
                sample = np.load(self.path_list[i])
                d = {'image': torch.from_numpy(sample['image']), "label": torch.from_numpy(sample['label']).long()}
            except:
                print(f"{self.path_list[i]} is bad! auto remove it.")
                os.remove(self.path_list[i])
                k = i
                i = random.randint(0, i - 1)
                self.path_list[k] = self.path_list[i]
        return d
