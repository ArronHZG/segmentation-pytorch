import os
import sys
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from experiments.datasets.open import get_segmentation_dataset
from experiments.datasets.open.base import decode_segmap
from experiments.datasets.path import Path
from experiments.utils.tools import make_sure_path_exists

# np.set_printoptions(threshold=np.inf)

plt.rcParams['savefig.dpi'] = 500  # 图片像素
plt.rcParams['figure.dpi'] = 500  # 分辨率


def test_data(dataset_name, **kwargs):
    test_path = make_sure_path_exists(os.path.join(Path.project_root, "dataset_test", dataset_name))

    dataset = get_segmentation_dataset(dataset_name, **kwargs)

    print(dataset)

    for sample, index in tqdm(zip(dataset, range(len(dataset)))):
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


if __name__ == '__main__':
    test_data("pascal_voc")
