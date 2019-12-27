import gc
import math
import os
import sys

import albumentations as A
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader

np.set_printoptions(threshold=sys.maxsize)
Image.MAX_IMAGE_PIXELS = None


class TestOneImage:
    '''
    一次测试一张大图
    使用训练集原图作为评价指标数据，验证测试方法的有效性
    测试时，选择多尺度滑窗，上下翻转，镜像翻转
    得到每一个像素的分类置信度，对于置信度低于阈值的像素值分类采用 相近点类别投票
    '''

    def __init__(self, num_classes, size, image_path, batch_size, num_workers):
        self.image_path = image_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.size = size
        self.stride = self.size / 4
        self.in_c = 4

        self.mean = (0.52891074, 0.38070734, 0.40119018, 0.36884733)
        self.std = (0.24007008, 0.23784, 0.22267079, 0.21865861)

        self.origin_image = None
        self.images_size = None
        self.output_image = None

        self._read_test_file()  # 读取图片并标准化

    def _read_test_file(self):
        image_pil = Image.open(self.image_path)
        image_np = np.array(image_pil)
        self.origin_image = A.Normalize(mean=self.mean, std=self.std, p=1)(image=image_np)['image']
        self.images_size = self.origin_image.shape
        self.output_image = np.zeros((self.images_size[0], self.images_size[1], self.num_classes))

    def get_DataLoader(self):
        dataset = TestDataSet(self.origin_image, self.size, self.stride)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=self.num_workers)

    def fill_image(self, sample):
        temp = sample['image']
        sample['image'] = temp.permute(0, 2, 3, 1).cpu().numpy()
        l = len(sample["x1"])
        for i in range(l):
            image = sample["image"][i]

            inside_x1 = int(image.shape[0] / 4)
            inside_y1 = int(image.shape[1] / 4)
            inside_x2 = int(image.shape[0] - image.shape[0] / 4)
            inside_y2 = int(image.shape[1] - image.shape[1] / 4)

            x1 = sample["x1"][i] + inside_x1
            y1 = sample["y1"][i] + inside_y1
            x2 = sample["x2"][i] - inside_x1
            y2 = sample["y2"][i] - inside_y1

            self.output_image[x1:x2, y1:y2, :] += image[inside_x1:inside_x2, inside_y1:inside_y2, :]

        del sample
        gc.collect()


class TestDataSet(data.Dataset):
    NUM_CLASSES = 16

    def __init__(self, image, crop_size, stride):
        super(TestDataSet, self).__init__()
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


def testFlip():
    num_classes = 4
    size = 1024
    name = 'GF2_PMS1__20150212_L1A0000647768-MSS1.tif'
    image_dir = '/home/arron/dataset/rssrai2019/train/image'
    save_dir = '/home/arron/dataset/rssrai2019/train/image-output'
    batch_size = 2
    num_workers = 4
    testOneImage = TestOneImage(num_classes, size, os.path.join(image_dir, name), batch_size, num_workers)

    i = 0
    for sample in testOneImage.get_DataLoader():
        i += 1
        print(i)
        testOneImage.fill_image(sample)
    output = testOneImage.output_image
    print(output.shape)
    # output /= np.max(output)
    # output *= 255
    # output = output.astype(np.uint8)
    # img = Image.fromarray(output)
    # os.makedirs(save_dir, exist_ok=True)
    # img.save(os.path.join(save_dir, name))
    output = torch.from_numpy(output).cuda()

    output = torch.argmax(output, -1)
    print(output)
    print(output.shape)


if __name__ == '__main__':
    testFlip()
