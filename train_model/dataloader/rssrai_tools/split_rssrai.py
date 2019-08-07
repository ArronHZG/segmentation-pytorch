import os
import random
import shutil
from glob import glob
from pprint import pprint

import numpy as np
from PIL import Image
from tqdm import tqdm
from train_model.config.mypath import Path


def split_image(image_path, image_name, save_path, mode, output_image_h_w=(256, 256)):
    '''
    读入图片为 H,W,C
    保存图片 (output_image_h_w,output_image_h_w,mode)
    :param image_path:
    :param image_name:
    :param save_path:
    :param mode:
    :return:
    '''
    save_name = image_name.split(".")[0]
    suffix = image_name.split(".")[1]
    file_image = Image.open(os.path.join(image_path, image_name))
    # print( tiff_image_file_image.mode )
    np_image = np.array(file_image)
    np_image_size = np_image.shape
    # print( np_image[0, 0, 0:] )
    split_image_size = None
    split_image_size = (output_image_h_w[0], output_image_h_w[1], 3)
    if mode == "CMYK":
        split_image_size = (output_image_h_w[0], output_image_h_w[1], 4)

    for h in range(np_image_size[0] // output_image_h_w[0]):
        for w in range(np_image_size[1] // output_image_h_w[1]):
            little_image = np_image[split_image_size[0] * h:split_image_size[0] * (h + 1),
                           split_image_size[1] * w:split_image_size[1] * (w + 1), :]
            assert little_image.shape == split_image_size
            save_image(little_image, save_path, f'{save_name}_{h}_{w}.{suffix}', mode=mode)

    # 当高不够时的边界保存
    if np_image_size[0] % output_image_h_w[0] != 0:
        # print("当高不够时的边界保存")
        h = np_image_size[0] // output_image_h_w[0]
        for w in range(np_image_size[1] // output_image_h_w[1]):
            little_image = np_image[-split_image_size[0]:,
                           split_image_size[1] * w:split_image_size[1] * (w + 1), :]
            assert little_image.shape == split_image_size
            save_image(little_image, save_path, f'{save_name}_{h}_{w}.{suffix}', mode=mode)

    # 当宽不够时的边界保存
    if np_image_size[1] % output_image_h_w[1] != 0:
        # print("当宽不够时的边界保存")
        w = np_image_size[1] // output_image_h_w[1]
        for h in range(np_image_size[0] // output_image_h_w[0]):
            little_image = np_image[split_image_size[0] * h:split_image_size[0] * (h + 1),
                           -split_image_size[1]:, :]
            assert little_image.shape == split_image_size
            save_image(little_image, save_path, f'{save_name}_{h}_{w}.{suffix}', mode=mode)

    # 保存左下角,三种情况
    if np_image_size[0] % output_image_h_w[0] != 0 and np_image_size[1] % output_image_h_w[1] != 0:
        h = np_image_size[0] // output_image_h_w[0]
        w = np_image_size[1] // output_image_h_w[1]
        little_image = np_image[-split_image_size[0]:, -split_image_size[1]:, :]
        assert little_image.shape == split_image_size
        save_image(little_image, save_path, f'{save_name}_{h}_{w}.{suffix}', mode=mode)

    if np_image_size[0] % output_image_h_w[0] == 0 and np_image_size[1] % output_image_h_w[1] != 0:
        h = np_image_size[0] // output_image_h_w[0] - 1
        w = np_image_size[1] // output_image_h_w[1]
        little_image = np_image[-split_image_size[0]:, -split_image_size[1]:, :]
        assert little_image.shape == split_image_size
        save_image(little_image, save_path, f'{save_name}_{h}_{w}.{suffix}', mode=mode)

    if np_image_size[0] % output_image_h_w[0] != 0 and np_image_size[1] % output_image_h_w[1] == 0:
        h = np_image_size[0] // output_image_h_w[0]
        w = np_image_size[1] // output_image_h_w[1] - 1
        little_image = np_image[-split_image_size[0]:, -split_image_size[1]:, :]
        assert little_image.shape == split_image_size
        save_image(little_image, save_path, f'{save_name}_{h}_{w}.{suffix}', mode=mode)


def save_image(np_image, path, name, mode="RGB"):
    image = Image.fromarray(np_image.astype('uint8'))
    image.mode = mode
    image.save(os.path.join(path, name))


def spilt_all_images(image_list, save_image_path, mode, output_image_h_w):
    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path)

    print(image_list)
    for image in tqdm(image_list):
        list = image.split("/")
        path = "/".join(list[:-1])
        name = list[-1]
        split_image(path, name, save_image_path, mode=mode, output_image_h_w=output_image_h_w)


def test_spilt_train_image():
    base_path = Path.db_root_dir("rssrai")

    # 图片
    image_path = os.path.join(base_path, "train", "img")
    image_list = glob(os.path.join(image_path, "*.tif"))
    save_image_path = os.path.join(base_path, "split_train", "img")
    spilt_all_images(image_list, save_image_path, mode="CMYK", output_image_h_w=(680, 720))

    # 标签
    label_path = os.path.join(base_path, "train", "label")
    label_list = glob(os.path.join(label_path, "*.tif"))
    save_label_path = os.path.join(base_path, "split_train", "label")
    spilt_all_images(label_list, save_label_path, mode="RGB", output_image_h_w=(680, 720))


def test_spilt_valid_image():
    # import pandas as pd
    # df = pd.read_csv("valid_set.csv")
    # name_list = df["文件名"].values.tolist()
    # print(name_list)

    base_path = Path.db_root_dir("rssrai")

    save_label_path = os.path.join(base_path, "split_valid_256", "label")
    if not os.path.exists(save_label_path):
        os.makedirs(save_label_path)
    label_path = os.path.join(base_path, "split_valid", "label")

    save_image_path = os.path.join(base_path, "split_valid_256", "img")
    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path)
    image_path = os.path.join(base_path, "split_valid", "img")

    label_name_list = [path_name.split("/")[-1] for path_name in glob(os.path.join(label_path, "*"))]

    print(len(label_name_list))

    for label_name in tqdm(label_name_list):
        image_name = label_name.replace("_label", "")
        # print(image_name)
        # print(label_name)
        split_image(label_path, label_name, save_label_path, mode="RGB")
        split_image(image_path, image_name, save_image_path, mode="CMYK")


def testOneImage():
    base_path = Path.db_root_dir("rssrai")
    path = '/home/arron/Documents/grey/Project_Rssrai/rssrai/train/img'
    name = 'GF2_PMS1__20150212_L1A0000647768-MSS1.tif'

    file_image = Image.open(os.path.join(path, name))

    np_image = np.array(file_image)[:, :, 1:]

    image = Image.fromarray(np_image.astype('uint8')).convert("RGB")
    image.save(os.path.join(base_path, name))

    # split_image( path, name, base_path, mode="CMYK" )


def testGetValid():
    base_path = Path.db_root_dir("rssrai")

    label_path = os.path.join(base_path, "split_train", "label")

    img_path = os.path.join(base_path, "split_train", "img")

    valid_label_path = os.path.join(base_path, "split_valid", "label")

    valid_img_path = os.path.join(base_path, "split_valid", "img")

    shutil.rmtree(valid_label_path)
    os.makedirs(valid_label_path)

    shutil.rmtree(valid_img_path)
    os.makedirs(valid_img_path)

    label_name_list = [path_name.split("/")[-1] for path_name in glob(os.path.join(label_path, "*"))]

    random.shuffle(label_name_list)

    print(len(label_name_list))

    valid_label_name_list = label_name_list[850:]

    pprint(len(valid_label_name_list))

    for label_name in valid_label_name_list:
        img_name = label_name.replace("_label", "")
        print(valid_label_path, label_name)
        print(valid_img_path, img_name)
        shutil.move(os.path.join(label_path, label_name), os.path.join(valid_label_path, label_name))
        shutil.move(os.path.join(img_path, img_name), os.path.join(valid_img_path, img_name))


if __name__ == '__main__':
    # test_spilt_train_image()
    # testGetValid()
    # test_spilt_valid_image()
    # pass
    li = glob('/home/arron/Documents/grey/Project_Rssrai/rssrai/split_valid_256/img/*')
    print(len(li))
    li = glob('/home/arron/Documents/grey/Project_Rssrai/rssrai/split_valid_256/label/*')
    print(len(li))
