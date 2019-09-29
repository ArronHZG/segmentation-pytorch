import os
import random
from glob import glob
from pprint import pprint

import albumentations as A
import matplotlib.pyplot  as plt
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader

from experiments.datasets.mypath import Path
from experiments.datasets.private.rssrai_tools import mean, std, encode_segmap, decode_segmap


class Xian(data.Dataset):
    NUM_CLASSES = 10

    def __init__(self, type='train', base_dir=Path.db_root_dir('rssrai')):

        assert type in ['train', 'valid', 'test']
        super().__init__()
        self._base_dir = base_dir
        self.type = type
        self.in_c = 4
        self.mean = mean
        self.std = std
        self.im_ids = []
        self.images = []
        self.categories = []

        # 加载数据
        if self.type == 'train':
            self._label_path_list = glob(os.path.join(self._base_dir, 'split_train', 'label', '*.tif'))
            self._label_name_list = [name.split('/')[-1] for name in self._label_path_list]
            self._image_dir = os.path.join(self._base_dir, 'split_train', 'img')
            self._label_dir = os.path.join(self._base_dir, 'split_train', 'label')

            self.len = len(self._label_path_list)

        if self.type == 'valid':
            self._label_path_list = glob(os.path.join(self._base_dir, 'split_valid_256', 'label', '*.tif'))
            self._label_name_list = [name.split('/')[-1] for name in self._label_path_list]
            self._image_dir = os.path.join(self._base_dir, 'split_valid_256', 'img')
            self._label_dir = os.path.join(self._base_dir, 'split_valid_256', 'label')

            self.len = len(self._label_name_list)

    def __getitem__(self, index):
        return self.transform(self.get_numpy_image(index))

    def __len__(self):
        return self.len
        # return 10

    def __str__(self):
        return 'Rssrai(split=' + str(self.type) + ')'

    def get_numpy_image(self, index):
        '''
        训练集随机选一张图片,然后随机crop
        验证集按顺序选取
        测试集按顺序选取
        '''
        sample = None
        if self.type == 'train':
            sample = self._read_file(self._label_name_list[index])
            sample = self._random_enhance(sample)
        if self.type == 'valid':
            sample = self._read_file(self._label_name_list[index])
            sample = self._valid_enhance(sample)
        return sample

    def _random_enhance(self, sample):
        compose = A.Compose([
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
        if self.type != "test":
            sample['label'] = torch.from_numpy(sample['label']).long()
        return sample


def testData():
    plt.rcParams['savefig.dpi'] = 500  # 图片像素
    plt.rcParams['figure.dpi'] = 500  # 分辨率

    test_path = os.path.join(Path().db_root_dir("rssrai"), "测试输出")
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    rssrai = Rssrai(type="train")
    for i in rssrai:
        pprint(i["image"].shape)
        pprint(i["label"].shape)
        break
    data_loader = DataLoader(rssrai, batch_size=4, shuffle=True, num_workers=4)

    for ii, sample in enumerate(data_loader):
        print(sample['image'].shape)
        sample['image'] = sample['image'][:, 1:, :, :]
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            tmp = gt[jj]
            segmap = decode_segmap(tmp)
            img_tmp *= rssrai.std[1:]
            img_tmp += rssrai.mean[1:]
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(121)
            plt.imshow(img_tmp)
            plt.subplot(122)
            plt.imshow(segmap)
            # with open( f"{test_path}/rssrai-{ii}-{jj}.txt", "w" ) as f:
            #     f.write( str( img_tmp ) )
            #     f.write( str( tmp ) )
            #     f.write( str( segmap ) )
            plt.savefig(f"{test_path}/rssrai-{ii}-{jj}.jpg")
            plt.close('all')

        if ii == 3:
            break

    plt.show(block=True)


# def test_encode():
#     from PIL import Image
#     image = Image.open(
#         '/home/arron/Documents/grey/Project_Rssrai/rssrai/split_valid/label/GF2_PMS1__20150212_L1A0000647768-MSS1_label_0_0_0_0.tif' )
#     image = np.array( image )
#     mask = encode_segmap( image )
#     for i in range( image.shape[1] ):
#         pprint( image[0, i] )
#         pprint( mask[0, i] )


if __name__ == '__main__':
    testData()
