import math
import os
import time
from collections import OrderedDict
from glob import glob
from pprint import pprint

import albumentations as A
import gc
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

    def __init__(self, image_name, image_path, save_path, batch_size, num_workers):
        self.image_name = image_name
        self.image_path = image_path
        self.save_path = save_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.in_c = 4
        self.num_classes = 16

        self.mean = (0.52891074, 0.38070734, 0.40119018, 0.36884733)
        self.std = (0.24007008, 0.23784, 0.22267079, 0.21865861)

        self.images = {"origin": None,
                       "vertical": None,
                       "horizontal": None}
        self.images_size = None
        self.output_image = None

        self._read_test_file()  # 读取图片并标准化
        self.flip_image()  # 上下翻转,左右翻转

        # 每种图片使用3种不同的滑窗方式，得到3个DataSet,测试结果共同填充到一个结果矩阵

    def _read_test_file(self):
        image_pil = Image.open(os.path.join(self.image_path, self.image_name))
        image_np = np.array(image_pil)
        self.images["origin"] = A.Normalize(mean=self.mean, std=self.std, p=1)(image=image_np)['image']
        self.images_size = self.images["origin"].shape

        self.output_image = np.zeros((self.images_size[0], self.images_size[1], self.num_classes))

    def flip_image(self):
        self.images["vertical"] = self.vertical(self.images["origin"])
        self.images["horizontal"] = self.horizontal(self.images["origin"])

    def get_slide_dataSet(self, image):
        for multiple in [1]:
            size = 256 * multiple
            stride = size / 4
            # stride = size
            rssraiSet = RssraiTest(image, size, stride)
            yield DataLoader(rssraiSet, batch_size=self.batch_size,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=self.num_workers)

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

    def fill_image(self, type, sample):
        temp = sample['image']
        image = temp.permute(0, 2, 3, 1).cpu().numpy()
        del temp
        gc.collect()
        if 'vertical' == type:
            sample['image'] = self.vertical(image)
        if 'horizontal' == type:
            sample['image'] = self.horizontal(image)
        if 'origin' == type:
            sample['image'] = image
        self._fill(sample)

    def _fill(self, sample):
        l = len(sample["x1"])
        for i in range(l):
            # print(self.output_image[sample["x1"][i]:sample["x2"][i], sample["y1"][i]:sample["y2"][i], :].shape)
            # print(sample['image'][i].permute(1,2,0).shape)
            self.output_image[sample["x1"][i]:sample["x2"][i], sample["y1"][i]:sample["y2"][i], :] = \
                sample['image'][i]
        del sample["x1"]
        del sample["x2"]
        del sample["y1"]
        del sample["y2"]
        del sample["image"]
        gc.collect()
        # time.sleep(3)

    def vertical(self, image):
        return A.VerticalFlip(p=1)(image=image)['image']

    def horizontal(self, image):
        return A.HorizontalFlip(p=1)(image=image)['image']

    def saveResultRGBImage(self):
        # output (B,C,H,W) to (B,H,W)
        # print(output.size())
        image = np.argmax(self.output_image, axis=2)
        image = self.decode_segmap(image)
        image = Image.fromarray(image.astype('uint8'))
        image.save(os.path.join(self.save_path, f"{self.image_name[:-4]}_label.tif"))


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
            x1 = self.image_size[0] - self.crop_size
            x2 = self.image_size[0]

        if x2 <= self.image_size[0] and y2 > self.image_size[1]:
            y1 = self.image_size[1] - self.crop_size
            y2 = self.image_size[1]

        if x2 > self.image_size[0] and y2 > self.image_size[1]:
            x1 = self.image_size[0] - self.crop_size
            x2 = self.image_size[0]
            y1 = self.image_size[1] - self.crop_size
            y2 = self.image_size[1]

        return {"x1": x1, "x2": x2, "y1": y1, "y2": y2, "image": self.image[x1:x2, y1:y2, :]}

    def transform(self, sample):
        sample["image"] = torch.from_numpy(sample["image"]).permute(2, 0, 1)
        return sample


def printSample(sample):
    print(f"CropSample:\nx1={sample['x1']}\nx2={sample['x2']}\ny1={sample['y1']}\ny2={sample['y2']}" \
          + f"\nimage.shape={sample['image'].shape}")


def save_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def testFlip():
    base_dir = Path.db_root_dir('rssrai')
    image_dir = os.path.join(base_dir, 'test')
    save_dir = save_path(os.path.join(base_dir, 'test_output'))

    _img_path_list = glob(os.path.join(image_dir, '*.tif'))
    img_name_list = [name.split('/')[-1] for name in _img_path_list]
    pprint(img_name_list)
    pprint(image_dir)

    rssraiImage = RssraiTestOneImage(img_name_list[0], image_dir, save_dir, 10, 4)

    type = "origin"
    # type = "vertical"
    # type = "horizontal"
    print(rssraiImage.images[type].shape)
    for dateSet in rssraiImage.get_slide_dataSet(rssraiImage.images[type]):
        print(dateSet)
        for i in dateSet:
            shape = i['image'].shape
            print(shape)
            # i['image'] = torch.zeros(shape[0], rssraiImage.num_classes, shape[2], shape[3]).cuda()
            # rssraiImage.fill_image(type, i)

    rssraiImage.saveResultRGBImage()


if __name__ == '__main__':
    testFlip()
