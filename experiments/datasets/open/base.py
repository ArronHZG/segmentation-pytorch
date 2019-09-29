###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import albumentations as A
import numpy as np

import torch
import torch.utils.data as data

__all__ = ['BaseDataset', 'test_batchify_fn', 'decode_segmap']


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
            A.Resize(self.crop_size+20,self.crop_size+20),
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


def get_labels(label_number):
    """
    :return: (19 , 3)
    """
    label_19 = np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])

    label_21 = np.array([
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128]])

    label_colors = {19: label_19, 21: label_21}
    return label_colors[label_number]


def decode_segmap(label_mask, label_number):
    """Decode segmentation class labels into a color image
        :param label_mask:
        :param label_number:
    """
    color_list = get_labels(label_number)
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, len(color_list)):
        r[label_mask == ll] = color_list[ll, 0]
        g[label_mask == ll] = color_list[ll, 1]
        b[label_mask == ll] = color_list[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb.astype(np.uint8)
