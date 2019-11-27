import os
from glob import glob

import albumentations as A
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from experiments.utils.iotools import make_sure_path_exists
from .rssrai_utils import mean, std, encode_segmap
from ...path import Path


def read_csv(path):
    with open(path, "r+") as read:
        lines = read.readlines()
    name_list = [x.split('\n')[0] for x in lines]
    return name_list


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
        self.valid_path = os.path.join(self._base_dir, f"valid_{self.crop_size}")
        self.train_numpy_path = os.path.join(self._base_dir, f"train_numpy_{self.crop_size}")
        self.valid_numpy_path = os.path.join(self._base_dir, f"valid_numpy_{self.crop_size}")
        make_sure_path_exists(self.train_numpy_path)

        # 加载数据
        if self.mode is 'train' and self.is_load_numpy is False:
            work_dir = os.path.join(self._base_dir, "split_680_720")
            path_name_list = read_csv(os.path.join(work_dir, "valid_set.csv"))
            self.name_list = [os.path.split(path_name)[-1] for path_name in path_name_list]
            self._image_dir = os.path.join(work_dir, 'image')
            self._label_dir = os.path.join(work_dir, 'label')
            self.len = len(self.name_list)

        if self.mode is 'train' and self.is_load_numpy is True:
            self.path_list = glob(os.path.join(self.train_numpy_path, '*.npz'))
            self.len = len(self.path_list)

        if self.mode is 'val':
            self.path_list = glob(os.path.join(self.valid_path, 'image', '*.tif'))
            self.name_list = [os.path.split(path_name)[-1] for path_name in self.path_list]
            self._label_dir = os.path.join(self.valid_path, 'label')
            self._image_dir = os.path.join(self.valid_path, 'image')
            self.len = len(self.path_list)

        if self.mode is 'val' and self.is_load_numpy is True:
            self.path_list = glob(os.path.join(self.valid_numpy_path, '*.npz'))
            self.len = len(self.path_list)

    def __getitem__(self, index, is_save=False):
        if self.is_load_numpy is True:
            return self.load_numpy(index)
        else:
            sample = self.transform(self.get_numpy_image(index), is_save=is_save)
            return sample

    def __len__(self):
        return self.len

    def __str__(self):
        return f"[Rssrai {self.mode}] num_classes:{self.NUM_CLASSES} len: {self.__len__()}"

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
        compose = A.Compose([
            # A.PadIfNeeded(self.base_size, self.base_size, p=1),
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
    def _read_file(self, name):
        image_pil = Image.open(os.path.join(self._image_dir, name))
        image_np = np.array(image_pil)

        label_pil = Image.open(os.path.join(self._label_dir, name))
        label_np = np.array(label_pil)
        label_mask = encode_segmap(label_np)

        return {'image': image_np, 'label': label_mask}

    # def _get_random_file_name(self):
    #     return random.choice(self.name_list)

    def transform(self, sample, is_save):
        if not is_save:
            sample['image'] = torch.from_numpy(sample['image']).permute(2, 0, 1)
            sample['label'] = torch.from_numpy(sample['label']).long()
        return sample

    def load_numpy(self, index):
        sample = np.load(self.path_list[index])
        return {'image': torch.from_numpy(sample['image']).permute(2, 0, 1),
                "label": torch.from_numpy(sample['label']).long()}
