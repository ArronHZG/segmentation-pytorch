import math
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


class RssraiTestOneImage():
    '''
    一次测试一张大图

    使用训练集原图作为评价指标数据，验证测试方法的有效性

    测试时，选择多尺度滑窗，上下翻转，镜像翻转

    得到每一个像素的分类置信度，对于置信度低于阈值的像素值分类采用 相近点类别投票

    '''

    def __init__(self, image_name, image_path, batch_size, num_workers):
        self.image_name = image_name
        self.image_path = image_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.in_c = 4

        self.mean = (0.52891074, 0.38070734, 0.40119018, 0.36884733)
        self.std = (0.24007008, 0.23784, 0.22267079, 0.21865861)

        self.image = None
        self.image_vertical_flip = None
        self.image_horizontal_flip = None

        self._read_test_file()  # 读取图片并标准化
        self._get_vertical_flip_image()  # 上下翻转
        self._get_horizontal_flip_image()  # 左右翻转

        # 每种图片使用3种不同的滑窗方式，得到3个DataSet,测试结果共同填充到一个结果矩阵

    def _read_test_file(self):
        image_pil = Image.open(os.path.join(self.image_path, self.image_name))
        image_np = np.array(image_pil)
        self.image = A.Normalize(mean=self.mean, std=self.std, p=1)(image=image_np)['image']

    def _get_vertical_flip_image(self):
        self.image_vertical_flip = A.VerticalFlip(p=1)(image=self.image)['image']

    def _get_horizontal_flip_image(self):
        self.image_vertical_flip = A.HorizontalFlip(p=1)(image=self.image)['image']

    def get_slide_dataSet(self, image):
        for multiple in [1, 2, 4]:
            size = 256 * multiple
            stride = size / 2
            yield RssraiTest(image, size, stride)
        # test_loader = DataLoader(a, batch_size=self.batch_size, shuffle=False, pin_memory=True,
        #                          num_workers=self.num_workers)

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


class RssraiTest(data.Dataset):
    NUM_CLASSES = 16

    def __init__(self, image, crop_size, stride):
        super(RssraiTest, self).__init__()
        self.image = image
        self.crop_size = crop_size
        self.stride = stride
        self.image_size = image.shape
        self.w_num = math.ceil((self.image_size[0] - crop_size) / stride + 1)  # 向上取整
        self.h_num = math.ceil((self.image_size[1] - crop_size) / stride + 1)  # 向上取整

    def __len__(self):
        return self.w_num * self.h_num

    def __getitem__(self, index):
        # w + h*w_num = index
        h = index // self.w_num
        w = index - h * self.w_num
        return self.transform(self.crop_image(w, h))

    def crop_image(self, w, h):
        # print(h)
        # print(w)
        x1 = int(w * self.stride)
        y1 = int(h * self.stride)

        x2 = int(x1 + self.crop_size)
        y2 = int(y1 + self.crop_size)

        # 异常处理
        if x2 > self.image_size[0] and y2 < self.image_size[1]:
            return self.image[-self.crop_size:, y1:y2, :]

        if x2 <= self.image_size[0] and y2 > self.image_size[1]:
            return self.image[x1:x2, -self.crop_size:, :]

        if x2 > self.image_size[0] and y2 > self.image_size[1]:
            return self.image[-self.crop_size:, -self.crop_size:, :]

        return self.image[x1:x2, y1:y2, :]

    def transform(self, image):
        return torch.from_numpy(image).permute(2, 0, 1)


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
            segmap = rssrai.decode_segmap(tmp).astype(np.uint8)
            img_tmp *= rssrai.std[1:]
            img_tmp += rssrai.mean[1:]
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(121)
            plt.imshow(img_tmp, vmin=0, vmax=255)
            plt.subplot(122)
            plt.imshow(segmap, vmin=0, vmax=255)

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


def testFlip():
    base_dir = Path.db_root_dir('rssrai')
    image_dir = os.path.join(base_dir, 'test')
    _img_path_list = glob(os.path.join(image_dir, '*.tif'))
    img_name_list = [name.split('/')[-1] for name in _img_path_list]
    lenth = len(img_name_list)
    pprint(img_name_list)
    pprint(image_dir)

    rssrai = RssraiTestOneImage(img_name_list[0], image_dir, 64, 4)
    print(rssrai.image_vertical_flip.shape)
    for dateSet in rssrai.get_slide_dataSet(rssrai.image):
        print(dateSet)
        for image in dateSet:
            # print(image.shape)
            pass
        print(image.shape)


if __name__ == '__main__':
    testFlip()
