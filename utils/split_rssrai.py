import os
from glob import glob
from pprint import pprint

import numpy as np
from PIL import Image
from tqdm import tqdm
from utils.mypath import Path


# def split_train_image(image_path, image_name, save_path, mode):
#     '''
#     读入图片为 H,W,C
#     :param image_path:
#     :param image_name:
#     :param save_path:
#     :param mode:
#     :return:
#     '''
#     save_name = image_name.split( "." )[0]
#     suffix = image_name.split( "." )[1]
#     file_image = Image.open( os.path.join( image_path, image_name ) )
#     # print( tiff_image_file_image.mode )
#     np_image = np.array( file_image )
#     # print( np_image.shape )
#     # print( np_image[0, 0, 0:] )
#     split_image_size = None
#     if mode == "CMYK":
#         split_image_size = (1700, 1800, 4)
#     elif mode == "RGB":
#         split_image_size = (1700, 1800, 3)
#     for h in range( 4 ):
#         for w in range( 4 ):
#             little_image = np_image[split_image_size[0] * h:split_image_size[0] * (h + 1),
#                            split_image_size[1] * w:split_image_size[1] * (w + 1), :]
#             assert little_image.shape == split_image_size
#             save_image( little_image, save_path, f'{save_name}_{h}_{w}.{suffix}', mode=mode )


# def testOne():
#     base_path = Path.db_root_dir( "rssrai" )
#     image_path = os.path.join( base_path, "train", "img", )
#     split_train_image( image_path, "GF2_PMS1__20150212_L1A0000647768-MSS1.tif", base_path, mode="CMYK" )


# def test_spilt_all_image():
#     base_path = Path.db_root_dir( "rssrai" )
#     save_path = os.path.join( base_path, "split_train", "img" )
#     if not os.path.exists( save_path ):
#         os.makedirs( save_path )
#     image_path = os.path.join( base_path, "train", "img", )
#     from glob import glob
#     image_list = glob( os.path.join( image_path, "*.tif" ) )
#     for image in tqdm( image_list ):
#         list = image.split( "/" )
#         path = "/".join( list[:-1] )
#         name = list[-1]
#         # print(path)
#         # print(name)
#         split_train_image( path, name, save_path, mode="CMYK" )


# def testOneLabel():
#     base_path = Path.db_root_dir( "rssrai" )
#     image_path = os.path.join( base_path, "split_train", "label", )
#     split_train_image( image_path, "GF2_PMS1__20150212_L1A0000647768-MSS1_label.tif", base_path, mode="RGB" )


# def test_spilt_all_label():
#     base_path = Path.db_root_dir( "rssrai" )
#     save_path = os.path.join( base_path, "split_train", "label" )
#     if not os.path.exists( save_path ):
#         os.makedirs( save_path )
#     image_path = os.path.join( base_path, "train", "label", )
#     from glob import glob
#     image_list = glob( os.path.join( image_path, "*.tif" ) )
#     print(image_list)
#     for image in tqdm( image_list ):
#         list = image.split( "/" )
#         path = "/".join( list[:-1] )
#         name = list[-1]
#         # print(path)
#         # print(name)
#         split_train_image( path, name, save_path, mode="RGB" )


def split_image(image_path, image_name, save_path, mode, output_image_h_w=(256, 256)):
    '''
    读入图片为 H,W,C
    :param image_path:
    :param image_name:
    :param save_path:
    :param mode:
    :return:
    '''
    save_name = image_name.split( "." )[0]
    suffix = image_name.split( "." )[1]
    file_image = Image.open( os.path.join( image_path, image_name ) )
    # print( tiff_image_file_image.mode )
    np_image = np.array( file_image )
    np_image_size = np_image.shape
    # print( np_image[0, 0, 0:] )
    split_image_size = None
    if mode == "CMYK":
        split_image_size = (output_image_h_w[0], output_image_h_w[1], 4)
    elif mode == "RGB":
        split_image_size = (output_image_h_w[0], output_image_h_w[1], 3)
    for h in range( np_image_size[0] // output_image_h_w[0] ):
        for w in range( np_image_size[1] // output_image_h_w[1] ):
            little_image = np_image[split_image_size[0] * h:split_image_size[0] * (h + 1),
                           split_image_size[1] * w:split_image_size[1] * (w + 1), :]
            assert little_image.shape == split_image_size
            save_image( little_image, save_path, f'{save_name}_{h}_{w}.{suffix}', mode=mode )

    # 当高不够时的边界保存
    if np_image_size[0] % output_image_h_w[0] != 0:
        print( "当高不够时的边界保存" )
        h = np_image_size[0] // output_image_h_w[0]
        for w in range( np_image_size[1] // output_image_h_w[1] ):
            little_image = np_image[-split_image_size[0]:,
                           split_image_size[1] * w:split_image_size[1] * (w + 1), :]
            assert little_image.shape == split_image_size
            save_image( little_image, save_path, f'{save_name}_{h}_{w}.{suffix}', mode=mode )

    # 当宽不够时的边界保存

    if np_image_size[1] % output_image_h_w[1] != 0:
        print( "当宽不够时的边界保存" )
        w = np_image_size[1] // output_image_h_w[1]
        for h in range( np_image_size[0] // output_image_h_w[0] ):
            little_image = np_image[split_image_size[0] * h:split_image_size[0] * (h + 1),
                           -split_image_size[1]:, :]
            assert little_image.shape == split_image_size
            save_image( little_image, save_path, f'{save_name}_{h}_{w}.{suffix}', mode=mode )

    # 保存左下角,三种情况
    if np_image_size[0] % output_image_h_w[0] != 0 and np_image_size[1] % output_image_h_w[1] != 0:
        h = np_image_size[0] // output_image_h_w[0]
        w = np_image_size[1] // output_image_h_w[1]
        little_image = np_image[-split_image_size[0]:, -split_image_size[1]:, :]
        assert little_image.shape == split_image_size
        save_image( little_image, save_path, f'{save_name}_{h}_{w}.{suffix}', mode=mode )

    if np_image_size[0] % output_image_h_w[0] == 0 and np_image_size[1] % output_image_h_w[1] != 0:
        h = np_image_size[0] // output_image_h_w[0] - 1
        w = np_image_size[1] // output_image_h_w[1]
        little_image = np_image[-split_image_size[0]:, -split_image_size[1]:, :]
        assert little_image.shape == split_image_size
        save_image( little_image, save_path, f'{save_name}_{h}_{w}.{suffix}', mode=mode )

    if np_image_size[0] % output_image_h_w[0] != 0 and np_image_size[1] % output_image_h_w[1] == 0:
        h = np_image_size[0] // output_image_h_w[0]
        w = np_image_size[1] // output_image_h_w[1] - 1
        little_image = np_image[-split_image_size[0]:, -split_image_size[1]:, :]
        assert little_image.shape == split_image_size
        save_image( little_image, save_path, f'{save_name}_{h}_{w}.{suffix}', mode=mode )


def save_image(np_image, path, name, mode="RGB"):
    image = Image.fromarray( np_image.astype( 'uint8' ) )
    image.mode = mode
    image.save( os.path.join( path, name ) )
    # tiff_image_file_image = Image.open( os.path.join( save_path, "1.tiff" ) )
    # print( tiff_image_file_image.mode )
    # np_image = np.array( tiff_image_file_image )
    # print( np_image.shape )
    # print( np_image[0, 0, 0:] )


def spilt_all_images(image_list, save_image_path, mode, output_image_h_w):
    if not os.path.exists( save_image_path ):
        os.makedirs( save_image_path )

    print( image_list )
    for image in tqdm( image_list ):
        list = image.split( "/" )
        path = "/".join( list[:-1] )
        name = list[-1]
        split_image( path, name, save_image_path, mode=mode, output_image_h_w=output_image_h_w )


def test_spilt_train_image():
    base_path = Path.db_root_dir( "rssrai" )

    # 图片
    image_path = os.path.join( base_path, "train", "img" )
    image_list = glob( os.path.join( image_path, "*.tif" ) )
    save_image_path = os.path.join( base_path, "split_train", "img" )
    spilt_all_images( image_list, save_image_path, mode="CMYK", output_image_h_w=(680, 720) )

    # 标签
    label_path = os.path.join( base_path, "train", "label" )
    label_list = glob( os.path.join( label_path, "*.tif" ) )
    save_label_path = os.path.join( base_path, "split_train", "label" )
    spilt_all_images( label_list, save_label_path, mode="RGB", output_image_h_w=(680, 720) )


def test_spilt_valid_image():
    base_path = Path.db_root_dir( "rssrai" )

    save_label_path = os.path.join( base_path, "split_valid", "label" )
    if not os.path.exists( save_label_path ):
        os.makedirs( save_label_path )
    image_label_path = os.path.join( base_path, "split_train", "label" )

    save_image_path = os.path.join( base_path, "split_valid", "img" )
    if not os.path.exists( save_image_path ):
        os.makedirs( save_image_path )
    image_path = os.path.join( base_path, "split_train", "img" )

    import pandas as pd
    df = pd.read_csv( "valid_set.csv" )
    name_list = df["文件名"].values.tolist()
    print(name_list)
    for label_name in tqdm( name_list ):
        image_name = label_name.replace( "_label", "" )
        split_image( image_label_path, label_name, save_label_path, mode="RGB" )
        split_image( image_path, image_name, save_image_path, mode="CMYK" )


if __name__ == '__main__':
    # test_spilt_train_image()
    test_spilt_valid_image()
