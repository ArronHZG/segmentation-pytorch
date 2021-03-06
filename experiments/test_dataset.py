# cython: language_level=3
import os

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.datasets import get_segmentation_dataset
from experiments.datasets.utils import decode_segmap
from experiments.utils.iotools import make_sure_path_exists

# np.set_printoptions(threshold=np.inf)

plt.rcParams['savefig.dpi'] = 500  # 图片像素
plt.rcParams['figure.dpi'] = 500  # 分辨率


def test_data(dataset_name, **kwargs):
    test_path = make_sure_path_exists(os.path.join(kwargs['basic_dir'], "dataset_test", dataset_name))

    dataset = get_segmentation_dataset(dataset_name, **kwargs)

    print(dataset)

    for sample, index in zip(dataset, range(len(dataset))):
        if len(sample["image"].shape) is 4:
            del sample["image"][0]
        # recover image
        img_tmp = sample['image'].permute(1, 2, 0).numpy()
        img_tmp *= dataset.std
        img_tmp += dataset.mean
        img_tmp *= 255.0
        img_tmp = img_tmp.astype(np.uint8)
        # print(f"{img_tmp.max()}  {img_tmp.min()}")
        # decode label
        gt = sample['label'].numpy()
        seg_map = decode_segmap(gt, dataset.NUM_CLASSES)
        # print(f"{seg_map.max()}  {seg_map.min()}")
        # show
        plt.figure()
        plt.title('display')
        plt.subplot(121)
        plt.imshow(img_tmp)
        plt.subplot(122)
        plt.imshow(seg_map)
        plt.savefig(os.path.join(test_path, f"{index}.jpg"))
        # plt.show(block=True)
        plt.close('all')

        if index is 10:
            break


def get_numpy(dataset_name, **kwargs):
    dataset = get_segmentation_dataset(dataset_name, **kwargs)

    train_loader = DataLoader(dataset,
                              batch_size=20,
                              pin_memory=True,
                              num_workers=8)
    print(len(train_loader))
    for sample in tqdm(train_loader):
        pass


if __name__ == '__main__':
    # test_data("rssrai", mode='train', is_load_numpy=True)
    # test_data("rssrai", mode='train')
    # test_data("rssrai", mode='val')
    basic_dir = '/home/arron/dataset/rssrai2019'

    get_numpy("rssrai", basic_dir=basic_dir, mode='train')
