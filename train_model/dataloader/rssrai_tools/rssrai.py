import os
import random
from collections import OrderedDict
from glob import glob
from pprint import pprint

import albumentations as A
import matplotlib.pyplot  as plt
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader

from train_model.config.mypath import Path

color_name_map = OrderedDict({(0, 200, 0): '水田',
                              (150, 250, 0): '水浇地',
                              (150, 200, 150): '旱耕地',
                              (200, 0, 200): '园地',
                              (150, 0, 250): '乔木林地',
                              (150, 150, 250): '灌木林地',
                              (250, 200, 0): '天然草地',
                              (200, 200, 0): '人工草地',
                              (200, 0, 0): '工业用地',
                              (250, 0, 150): '城市住宅',
                              (200, 150, 150): '村镇住宅',
                              (250, 150, 150): '交通运输',
                              (0, 0, 200): '河流',
                              (0, 150, 200): '湖泊',
                              (0, 200, 250): '坑塘',
                              (0, 0, 0): '其他类别'})

color_index_map = OrderedDict({(0, 200, 0): 0,
                               (150, 250, 0): 1,
                               (150, 200, 150): 2,
                               (200, 0, 200): 3,
                               (150, 0, 250): 4,
                               (150, 150, 250): 5,
                               (250, 200, 0): 6,
                               (200, 200, 0): 7,
                               (200, 0, 0): 8,
                               (250, 0, 150): 9,
                               (200, 150, 150): 10,
                               (250, 150, 150): 11,
                               (0, 0, 200): 12,
                               (0, 150, 200): 13,
                               (0, 200, 250): 14,
                               (0, 0, 0): 15})

color_list = np.array([[0, 200, 0],
                       [150, 250, 0],
                       [150, 200, 150],
                       [200, 0, 200],
                       [150, 0, 250],
                       [150, 150, 250],
                       [250, 200, 0],
                       [200, 200, 0],
                       [200, 0, 0],
                       [250, 0, 150],
                       [200, 150, 150],
                       [250, 150, 150],
                       [0, 0, 200],
                       [0, 150, 200],
                       [0, 200, 250],
                       [0, 0, 0]])


class Rssrai(data.Dataset):
    NUM_CLASSES = 16

    def __init__(self, type='train', base_size=(512, 512), crop_size=(256, 256), base_dir=Path.db_root_dir('rssrai')):

        assert type in ['train', 'valid', 'test']
        super().__init__()
        self._base_dir = base_dir
        self.type = type
        self.in_c = 4
        self.mean = (0.52891074, 0.38070734, 0.40119018, 0.36884733)
        self.std = (0.24007008, 0.23784, 0.22267079, 0.21865861)
        self.crop_size = crop_size
        self.base_size = base_size
        self.im_ids = []
        self.images = []
        self.categories = []

        # 加载数据
        if self.type == 'train':
            # train_csv = os.path.join(self._base_dir, 'train_set.csv')
            # self._label_name_list = pd.read_csv(train_csv)["文件名"].values.tolist()
            self._label_path_list = glob(os.path.join(self._base_dir, 'split_train', 'label', '*.tif'))
            # print(self._label_path_list)
            self._label_name_list = [name.split('/')[-1] for name in self._label_path_list]
            # print(self._label_name_list)
            self._image_dir = os.path.join(self._base_dir, 'split_train', 'img')
            self._label_dir = os.path.join(self._base_dir, 'split_train', 'label')

            self.len = 14000

        if self.type == 'valid':
            self._label_path_list = glob(os.path.join(self._base_dir, 'split_valid_256', 'label', '*.tif'))
            self._label_name_list = [name.split('/')[-1] for name in self._label_path_list]
            self._image_dir = os.path.join(self._base_dir, 'split_valid_256', 'img')
            self._label_dir = os.path.join(self._base_dir, 'split_valid_256', 'label')
            # self._label_name_list = pd.read_csv( valid_csv )["文件名"].values.tolist()

            self.len = len(self._label_name_list)

        if self.type == 'test':
            self._img_path_list = glob(os.path.join(self._base_dir, 'split_test_256', 'img', '*.tif'))
            self._img_name_list = [name.split('/')[-1] for name in self._img_path_list]
            self._image_dir = os.path.join(self._base_dir, 'split_test_256', 'img')
            self.len = len(self._img_path_list)

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
            name = self._get_random_file_name()
            sample = self._read_file(name)
            sample = self._random_crop_and_enhance(sample)
        if self.type == 'valid':
            sample = self._read_file(self._label_name_list[index])
            sample = self._valid_enhance(sample)
        if self.type == 'test':
            sample = self._read_test_file(self._img_name_list[index])
            sample = self._test_enhance(sample)
        # sample["image"] = sample["image"][:, :, 1:]
        return sample

    def _random_crop_and_enhance(self, sample):
        compose = A.Compose([
            A.PadIfNeeded(self.base_size[0], self.base_size[1], p=1),
            # A.RandomSizedCrop((512,512),self.crop_size[0], self.crop_size[1], p=1),
            A.RandomCrop(self.crop_size[0], self.crop_size[1], p=1),
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
            A.PadIfNeeded(self.base_size[0], self.base_size[1], p=1),
            A.CenterCrop(self.crop_size[0], self.crop_size[1], p=1),
            A.Normalize(mean=self.mean, std=self.std, p=1)
        ], additional_targets={'image': 'image', 'label': 'mask'})
        return compose(**sample)

    def _test_enhance(self, sample):
        compose = A.Compose([
            A.PadIfNeeded(self.base_size[0], self.base_size[1], p=1),
            A.CenterCrop(self.crop_size[0], self.crop_size[1], p=1),
            A.Normalize(mean=self.mean, std=self.std, p=1)
        ], additional_targets={'image': 'image'})
        sample['image'] = compose(image=sample["image"])['image']
        return sample

    # @functools.lru_cache( maxsize=None )
    def _read_file(self, label_name):
        image_name = label_name.replace("_label", "")
        image_pil = Image.open(os.path.join(self._image_dir, image_name))
        image_np = np.array(image_pil)

        label_pil = Image.open(os.path.join(self._label_dir, label_name))
        label_np = np.array(label_pil)
        label_mask = self.encode_segmap(label_np)

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

    def decode_seg_map_sequence(self, label_masks):
        rgb_masks = []
        for label_mask in label_masks:
            rgb_mask = self.decode_segmap(label_mask)
            rgb_masks.append(rgb_mask)
        rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
        return rgb_masks

    def decode_segmap(self, label_mask):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
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
        return rgb

    def encode_segmap(self, label_image):
        """Encode segmentation label images as pascal classes
        Args:
            label_image (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        label_image = label_image.astype(int)
        label_mask = np.zeros((label_image.shape[0], label_image.shape[1]), dtype=np.int16)
        for ii, label in enumerate(color_list):
            label_mask[np.where(np.all(label_image == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask


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
            segmap = rssrai.decode_segmap(tmp)
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
