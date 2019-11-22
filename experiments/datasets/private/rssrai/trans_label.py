import os
import sys

import numpy as np
from PIL import Image

from experiments.datasets.path import Path
from experiments.datasets.private.rssrai.rssrai_utils import encode_segmap

np.set_printoptions(threshold=sys.maxsize)

Image.MAX_IMAGE_PIXELS = None


def encode_one_label(label_dir, label_name):
    label_pil = Image.open(os.path.join(label_dir, label_name))
    label_np = np.array(label_pil)
    label_mask = encode_segmap(label_np)
    return label_mask


if __name__ == '__main__':
    basic_dir = Path.db_root_dir("rssrai")
    label_vis_dir = os.path.join(basic_dir, "train", "label_vis")
    label_dir = os.path.join(basic_dir, "train", "label")
    name_list = [
        'GF2_PMS1__20150212_L1A0000647768-MSS1.tif',
        'GF2_PMS1__20150902_L1A0001015649-MSS1.tif',
        'GF2_PMS1__20151203_L1A0001217916-MSS1.tif',
        'GF2_PMS1__20160327_L1A0001491417-MSS1.tif',
        'GF2_PMS1__20160421_L1A0001537716-MSS1.tif',
        'GF2_PMS1__20160816_L1A0001765570-MSS1.tif',
        'GF2_PMS1__20160827_L1A0001793003-MSS1.tif',
        'GF2_PMS2__20150217_L1A0000658637-MSS2.tif',
        'GF2_PMS2__20160225_L1A0001433318-MSS2.tif',
        'GF2_PMS2__20160510_L1A0001573999-MSS2.tif'
    ]
    for name in name_list:
        print(name)
        label_mask = encode_one_label(label_vis_dir, name)
        # label_mask = np.random.randint(150, 255, (300, 300))
        print(label_mask.shape)
        label_img = Image.fromarray(label_mask, mode='L')
        label_img.save(os.path.join(label_dir, name))
