import glob
import os
import os.path as osp
import random
import sys
import time

import numpy as np
import torch
from PIL import Image

from experiments.datasets.private.rssrai.rssrai_utils import encode_segmap

np.set_printoptions(threshold=sys.maxsize)
Image.MAX_IMAGE_PIXELS = None


def func_time(func):
    def inner(*args, **kw):
        start_time = time.time()
        func(*args, **kw)
        end_time = time.time()
        print('run time：', end_time - start_time, 's')

    return inner


def make_sure_path_exists(some_dir):
    if not os.path.exists(some_dir):
        os.makedirs(some_dir)


def split_image(image_path, image_name, save_path, output_image_h_w=(512, 512)):
    make_sure_path_exists(save_path)
    file_image = Image.open(os.path.join(image_path, image_name))
    mode = file_image.mode
    torch_image = torch.from_numpy(np.array(file_image)).cuda()
    np_image_size = torch_image.shape
    code, suffix = os.path.splitext(image_name)
    print(f"\t torch_image,shape {torch_image.shape}")

    split_image_size = None
    if mode is "CMYK":
        split_image_size = (output_image_h_w[0], output_image_h_w[1], 4)
    elif mode is "RGB":
        split_image_size = (output_image_h_w[0], output_image_h_w[1], 3)
    elif mode is "L":
        split_image_size = (output_image_h_w[0], output_image_h_w[1])

    for h in range(np_image_size[0] // output_image_h_w[0]):
        for w in range(np_image_size[1] // output_image_h_w[1]):
            little_image = torch_image[split_image_size[0] * h:split_image_size[0] * (h + 1),
                           split_image_size[1] * w:split_image_size[1] * (w + 1)]
            assert little_image.shape == split_image_size
            save_image(little_image.cpu().numpy(), save_path, f'{code}_{h}_{w}{suffix}', mode=mode)

    if np_image_size[0] % output_image_h_w[0] != 0:
        h = np_image_size[0] // output_image_h_w[0]
        for w in range(np_image_size[1] // output_image_h_w[1]):
            little_image = torch_image[-split_image_size[0]:,
                           split_image_size[1] * w:split_image_size[1] * (w + 1)]
            assert little_image.shape == split_image_size
            save_image(little_image.cpu().numpy(), save_path, f'{code}_{h}_{w}{suffix}', mode=mode)

    if np_image_size[1] % output_image_h_w[1] != 0:
        w = np_image_size[1] // output_image_h_w[1]
        for h in range(np_image_size[0] // output_image_h_w[0]):
            little_image = torch_image[split_image_size[0] * h:split_image_size[0] * (h + 1),
                           -split_image_size[1]:]
            assert little_image.shape == split_image_size
            save_image(little_image.cpu().numpy(), save_path, f'{code}_{h}_{w}{suffix}', mode=mode)

    if np_image_size[0] % output_image_h_w[0] != 0 and np_image_size[1] % output_image_h_w[1] != 0:
        h = np_image_size[0] // output_image_h_w[0]
        w = np_image_size[1] // output_image_h_w[1]
        little_image = torch_image[-split_image_size[0]:, -split_image_size[1]:]
        assert little_image.shape == split_image_size
        save_image(little_image.cpu().numpy(), save_path, f'{code}_{h}_{w}{suffix}', mode=mode)

    if np_image_size[0] % output_image_h_w[0] == 0 and np_image_size[1] % output_image_h_w[1] != 0:
        h = np_image_size[0] // output_image_h_w[0] - 1
        w = np_image_size[1] // output_image_h_w[1]
        little_image = torch_image[-split_image_size[0]:, -split_image_size[1]:]
        assert little_image.shape == split_image_size
        save_image(little_image.cpu().numpy(), save_path, f'{code}_{h}_{w}{suffix}', mode=mode)

    if np_image_size[0] % output_image_h_w[0] != 0 and np_image_size[1] % output_image_h_w[1] == 0:
        h = np_image_size[0] // output_image_h_w[0]
        w = np_image_size[1] // output_image_h_w[1] - 1
        little_image = torch_image[-split_image_size[0]:, -split_image_size[1]:]
        assert little_image.shape == split_image_size
        save_image(little_image.cpu().numpy(), save_path, f'{code}_{h}_{w}{suffix}', mode=mode)


def save_image(np_image, path, name, mode):
    image = Image.fromarray(np_image.astype('uint8'))
    image.mode = mode
    image.save(os.path.join(path, name))


def read_csv(path):
    with open(path, "r+") as read:
        lines = read.readlines()
    name_list = [x.split('\n')[0] for x in lines]
    return name_list


@func_time
def split_dataset(basic_dir):
    ford = "source_data"
    pixel = (680, 720)
    path_name_list = glob.glob(os.path.join(basic_dir, ford, "image", "*.tif"))
    name_list = [osp.split(path_name)[-1] for path_name in path_name_list]
    print("start")
    for index, name in enumerate(name_list):
        print(f"{index + 1}/{len(name_list)}   {name}")
        types = ("image", "label", "label_vis")
        for type in types:
            split_image(osp.join(basic_dir, ford, type),
                        name,
                        osp.join(basic_dir, f"split_{pixel[0]}_{pixel[1]}", type),
                        output_image_h_w=pixel)
    print("end")


@func_time
def split_valid(basic_dir, pixel):
    work_dir = os.path.join(basic_dir, "split_680_720")
    path_name_list = read_csv(os.path.join(work_dir, "valid_set.csv"))
    name_list = [osp.split(path_name)[-1] for path_name in path_name_list]
    pixel_tuple = (pixel, pixel)
    print("start")
    for index, name in enumerate(name_list):
        print(f"{index + 1}/{len(name_list)}   {name}")
        types = ("image", "label")
        for type in types:
            split_image(osp.join(work_dir, type),
                        name,
                        osp.join(basic_dir, f"valid_{pixel_tuple[0]}", type),
                        output_image_h_w=pixel_tuple)
    print("end")


def split_train_and_valid(basic_dir):
    work_dir = os.path.join(basic_dir, "split_680_720")
    path_list = glob.glob(os.path.join(work_dir, "image", "*.tif"))
    train_f = open(os.path.join(work_dir, "train_set.csv"), "w+")
    valid_f = open(os.path.join(work_dir, "valid_set.csv"), "w+")

    for name in path_list:
        if random.randint(0, 4) is 0:
            valid_f.write(f"{name}\n")
        else:
            train_f.write(f"{name}\n")

    train_f.close()
    valid_f.close()

    with open(os.path.join(work_dir, "train_set.csv"), "r+") as read:
        lines = read.readlines()
        name_list = [x.split('\n')[0] for x in lines]
        print(len(name_list))


def transform_label_vis(basic_dir):
    ford = "source_data"
    save_path = os.path.join(basic_dir, ford, "label")
    make_sure_path_exists(save_path)
    path_name_list = glob.glob(os.path.join(basic_dir, ford, "label_vis", "*.tif"))
    name_list = [osp.split(path_name)[-1] for path_name in path_name_list]
    print("start")
    for index, name in enumerate(name_list):
        print(f"{index + 1}/{len(name_list)}   {name}")
        file_image = Image.open(osp.join(basic_dir, ford, "label_vis", name))
        label_np = np.array(file_image)
        label_mask = encode_segmap(label_np)
        save_image(label_mask, save_path, name, "L")
    print("end")


if __name__ == '__main__':
    basic_dir = '/home/deamov/dataset/rssrai2019'
    # transform_label_vis(basic_dir)
    split_dataset(basic_dir)
    # split_train_and_valid(basic_dir)
    # split_valid(basic_dir, 256)
    # split_valid(basic_dir, 512)
# p = 1 run time： 32.33511567115784 s
# p = 2 run time： 62.33348083496094 s
# p = 4 run time： 76.42453122138977 s
# p = 6 run time： 70.79261445999146 s
