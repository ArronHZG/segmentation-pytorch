# cython: language_level=3
###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import albumentations as A
import numpy as np

import torch
import torch.utils.data as data

__all__ = ['BaseDataset', 'test_batchify_fn']


class BaseDataset(data.Dataset):
    def __init__(self, root, split, mode=None, base_size=520, crop_size=480):
        self.root = root
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.in_c = 3
        # print(f'\nBaseDataset: base_size {base_size}, crop_size {crop_size}')

    def __getitem__(self, index):
        raise NotImplemented

    @property
    def num_class(self):
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        raise NotImplemented

    def make_pred(self, x):
        return x + self.pred_offset

    def _val_sync_transform(self, sample):
        compose = A.Compose([
            A.PadIfNeeded(self.base_size, self.base_size, p=1),
            A.CenterCrop(self.crop_size, self.crop_size, p=1),
            A.Normalize(),
        ], additional_targets={'image': 'image', 'label': 'mask'})
        sample = compose(**sample)

        sample["image"] = self._transform(sample["image"])
        sample["label"] = self._mask_transform(sample["label"])
        return sample

    def _sync_transform(self, sample):
        compose = A.Compose([
            A.PadIfNeeded(self.base_size, self.base_size, p=1),
            # A.RandomSizedCrop((self.crop_size - 10, self.crop_size + 10), self.crop_size, self.crop_size, p=1),
            A.Resize(self.crop_size + 20, self.crop_size + 20),
            A.CenterCrop(self.crop_size, self.crop_size, p=1),
            A.HorizontalFlip(),
            A.Blur(p=0.3),
            A.GaussNoise(p=0.3),
            A.Normalize(),
        ], additional_targets={'image': 'image', 'label': 'mask'})
        sample = compose(**sample)

        sample["image"] = self._transform(sample["image"])
        sample["label"] = self._mask_transform(sample["label"])
        return sample

    def _transform(self, img):
        return torch.from_numpy(img).permute(2, 0, 1)

    def _mask_transform(self, mask):
        return torch.from_numpy(mask).long()


def test_batchify_fn(feature):
    error_msg = "batch must contain tensors, tuples or lists; found {}"
    if isinstance(feature[0], (str, torch.Tensor)):
        return list(feature)
    elif isinstance(feature[0], (tuple, list)):
        feature = zip(*feature)
        return [test_batchify_fn(i) for i in feature]
    raise TypeError((error_msg.format(type(feature[0]))))

