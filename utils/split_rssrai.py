import os
from pprint import pprint

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.mypath import Path
import re
import unittest


def split_image(image_path, image_name, save_path, mode):
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
    # print( np_image.shape )
    # print( np_image[0, 0, 0:] )
    split_image_size = None
    if mode == "CMYK":
        split_image_size = (1700, 1800, 4)
    elif mode == "RGB":
        split_image_size = (1700, 1800, 3)
    for h in range( 4 ):
        for w in range( 4 ):
            little_image = np_image[split_image_size[0] * h:split_image_size[0] * (h + 1),
                           split_image_size[1] * w:split_image_size[1] * (w + 1),
                           :]
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


def testOne():
    base_path = Path.db_root_dir( "rssrai" )
    image_path = os.path.join( base_path, "train", "img", )
    split_image( image_path, "GF2_PMS1__20150212_L1A0000647768-MSS1.tif", base_path, mode="CMYK" )


def test_spilt_all_image():
    base_path = Path.db_root_dir( "rssrai" )
    save_path = os.path.join( base_path, "split", "train" )
    if not os.path.exists( save_path ):
        os.makedirs( save_path )
    image_path = os.path.join( base_path, "train", "img", )
    from glob import glob
    image_list = glob( os.path.join( image_path, "*.tif" ) )
    for image in tqdm( image_list ):
        list = image.split( "/" )
        path = "/".join( list[:-1] )
        name = list[-1]
        # print(path)
        # print(name)
        split_image( path, name, save_path, mode="CMYK" )


def testOneLabel():
    base_path = Path.db_root_dir( "rssrai" )
    image_path = os.path.join( base_path, "train", "label", )
    split_image( image_path, "GF2_PMS1__20150212_L1A0000647768-MSS1_label.tif", base_path, mode="RGB" )


def test_spilt_all_label():
    base_path = Path.db_root_dir( "rssrai" )
    save_path = os.path.join( base_path, "split", "label" )
    if not os.path.exists( save_path ):
        os.makedirs( save_path )
    image_path = os.path.join( base_path, "train", "label", )
    from glob import glob
    image_list = glob( os.path.join( image_path, "*.tif" ) )
    for image in tqdm( image_list ):
        list = image.split( "/" )
        path = "/".join( list[:-1] )
        name = list[-1]
        # print(path)
        # print(name)
        split_image( path, name, save_path, mode="RGB" )


if __name__ == '__main__':
    test_spilt_all_image()
    test_spilt_all_label()
