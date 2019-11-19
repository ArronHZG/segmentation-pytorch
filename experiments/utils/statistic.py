import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

from experiments.datasets.path import Path
from experiments.datasets.utils import make_data_loader
from experiments.utils.iotools import make_sure_path_exists

plt.rcParams['savefig.dpi'] = 200  # 图片像素
plt.rcParams['figure.dpi'] = 200  # 分辨率


def statistic_log(f, name, all_number, all_number_ignore_bg, each_number):
    f.write(f'\n{name}:\n')
    f.write(f'\t number: \t\t\t\t{[f"{x:.4f}" for x in list(each_number)]}:\n')
    f.write(f'\t percentage: \t\t\t{[f"{x:.4f}" for x in list(each_number / all_number)]}:\n')
    f.write(
        f'\t percentage_ignore_bg: \t{" " * 10}{[f"{x:.4f}" for x in list(each_number[1:] / all_number_ignore_bg)]}:\n')


def statistic_one_label(f, label_dir, statistic_dir, name, num_classes):
    class_list = [x for x in range(1, num_classes)]
    label_pil = Image.open(os.path.join(label_dir, name))
    label_array = np.array(label_pil).reshape((1, -1))
    each_number, bins = np.histogram(label_array, bins=num_classes)
    each_number.reshape((1, num_classes))
    all_number = each_number.sum()
    all_number_ignore_bg = each_number[1:].sum()
    statistic_log(f, name, all_number, all_number_ignore_bg, each_number)
    code, _ = os.path.splitext(name)
    sns_barplot(class_list, each_number, all_number_ignore_bg, statistic_dir, f"{code}_statistic.png")
    return each_number, all_number, all_number_ignore_bg


def sns_barplot(class_list, each_number, all_number_ignore_bg, statistic_dir, file_name):
    sns.set_style('darkgrid')
    bp = sns.barplot(x=class_list, y=each_number[1:] / all_number_ignore_bg)

    # # 在柱状图的上面显示各个类别的数量
    # for index, percentage in enumerate(list(each_number[1:] / all_number_ignore_bg)):
    #     # 在柱状图上绘制该类别的数量
    #     bp.text(index, percentage, round(percentage, 4), color="black", ha="center")

    plt.savefig(os.path.join(statistic_dir, file_name))
    plt.cla()


def statistic(dataset_name):
    basic_dir = Path.db_root_dir(dataset_name)
    _, _, num_classes = make_data_loader(dataset_name, base_size=512, crop_size=512)
    label_dir = os.path.join(basic_dir, "train", "label")
    statistic_dir = os.path.join(basic_dir, "train", "statistic")
    make_sure_path_exists(statistic_dir)
    path_name_list = glob.glob(os.path.join(label_dir, "*.tif"))
    print(f"label file number : {len(path_name_list)}")
    print(f"num_classes : {num_classes}")

    class_list = [x for x in range(1, num_classes)]
    sum_each_number = np.zeros(num_classes)
    sum_all_number = 0
    sum_all_number_ignore_bg = 0
    with open(os.path.join(statistic_dir, "label_statistic.txt"), "w+") as f:
        for path_name in path_name_list:
            dirt, name = os.path.split(path_name)
            print(name)
            each_number, all_number, all_number_ignore_bg = statistic_one_label(f, label_dir, statistic_dir, name,
                                                                                num_classes)
            sum_each_number += each_number
            sum_all_number += all_number
            sum_all_number_ignore_bg += all_number_ignore_bg

        f.write(f'\n\n===================================================================\n')
        statistic_log(f, "total", sum_all_number, sum_all_number_ignore_bg, sum_each_number)
        sns_barplot(class_list, sum_each_number, sum_all_number_ignore_bg, statistic_dir, "all_statistic.png")
