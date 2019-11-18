import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

from experiments.datasets.path import Path
from experiments.utils.iotools import make_sure_path_exists

plt.rcParams['savefig.dpi'] = 200  # 图片像素
plt.rcParams['figure.dpi'] = 200  # 分辨率

NUM_CLASSES = 16


def statistic_one_label(sum_each_number, sum_all_number, sum_all_number_ignore_bg):
    label_pil = Image.open(os.path.join(label_dir, name))
    label_array = np.array(label_pil).reshape((1, -1))
    each_number, _ = np.histogram(label_array, bins=NUM_CLASSES)
    each_number.reshape((1, NUM_CLASSES))
    sum_each_number += each_number
    all_number = each_number.sum()
    sum_all_number += all_number
    all_number_ignore_bg = each_number[1:].sum()
    sum_all_number_ignore_bg += all_number_ignore_bg
    f.write(f'\n{name}:\n')
    f.write(f'\t number: \t\t\t\t{[f"{x:.4f}" for x in list(each_number)]}:\n')
    f.write(f'\t percentage: \t\t\t{[f"{x:.4f}" for x in list(each_number / all_number)]}:\n')
    f.write(
        f'\t percentage_ignore_bg: \t{" " * 10}{[f"{x:.4f}" for x in list(each_number[1:] / all_number_ignore_bg)]}:\n')
    sns.set_style('darkgrid')
    sns.barplot(x=class_list, y=each_number[1:] / all_number_ignore_bg)
    code, _ = os.path.splitext(name)
    plt.savefig(os.path.join(statistic_dir, f"{code}_statistic.png"))
    plt.cla()
    return sum_all_number, sum_all_number_ignore_bg, sum_each_number


if __name__ == '__main__':

    basic_dir = Path.db_root_dir("rssrai")
    label_dir = os.path.join(basic_dir, "train", "label")
    statistic_dir = os.path.join(basic_dir, "train", "statistic")
    make_sure_path_exists(statistic_dir)

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
    class_list = [x for x in range(1, NUM_CLASSES)]
    sum_each_number = np.zeros(NUM_CLASSES)
    sum_all_number = 0
    sum_all_number_ignore_bg = 0
    with open(os.path.join(statistic_dir, "label_statistic.txt"), "w+") as f:
        for name in name_list:
            print(name)
            sum_all_number, sum_all_number_ignore_bg, sum_each_number = statistic_one_label(sum_all_number,
                                                                                            sum_all_number_ignore_bg,
                                                                                            sum_each_number)
        f.write(f'\n\n===================================================================\n')
        f.write(f'total:')
        f.write(f'\t number: \t\t\t{[f"{x:.4f}" for x in list(sum_each_number)]}:\n')
        f.write(f'\t percentage: \t\t\t{[f"{x:.4f}" for x in list(sum_each_number / sum_all_number)]}:\n')
        f.write(
            f'\t percentage_ignore_bg: \t{" " * 10}{[f"{x:.4f}" for x in list(sum_each_number[1:] / sum_all_number_ignore_bg)]}:\n')

        sns.barplot(x=class_list, y=sum_each_number[1:] / sum_all_number_ignore_bg)
        plt.savefig(os.path.join(statistic_dir, "all_statistic.png"))
        plt.cla()
