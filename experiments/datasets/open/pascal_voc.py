import os

import numpy as np
from PIL import Image

from .base import BaseDataset
from ..path import Path


class VOCSegmentation(BaseDataset):
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor', 'ambigious'
    ]
    NUM_CLASSES = 21

    def __init__(self, root=Path.db_root_dir("pascal_voc"),
                 split='train',
                 mode=None,
                 **kwargs):
        super(VOCSegmentation, self).__init__(root, split, mode, **kwargs)
        _voc_root = self.root
        _mask_dir = os.path.join(_voc_root, 'SegmentationClass')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')
        if self.split == 'train':
            _split_f = os.path.join(_splits_dir, 'trainval.txt')
        elif self.split == 'val':
            _split_f = os.path.join(_splits_dir, 'val.txt')
        elif self.split == 'test':
            _split_f = os.path.join(_splits_dir, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split.')
        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n') + ".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                if self.mode != 'test':
                    _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".png")
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        if self.mode != 'test':
            assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self.transform(np.array(img).astype("int32"))
            return {'image': img, 'label': None}
        target = Image.open(self.masks[index])
        sample = {'image': np.array(img).astype(np.uint8), 'label': np.array(target).astype(np.uint8)}
        # synchrosized transform
        if self.mode == 'train':
            return self._sync_transform(sample)
        elif self.mode == 'val':
            return self._val_sync_transform(sample)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.images)

    def __str__(self):
        return f"[VOC {self.mode}] num_classes:{self.NUM_CLASSES} len: {self.__len__()}"

    @property
    def pred_offset(self):
        return 0
