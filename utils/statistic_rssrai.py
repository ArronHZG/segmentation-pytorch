import os
from collections import OrderedDict
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm

from utils.mypath import Path

from multiprocessing import Pool

color_name_map = OrderedDict( {(0, 200, 0): '水田',
                               (150, 250, 0): '水浇地',
                               (150, 200, 150): '旱耕地',
                               (200, 0, 200): '园地',
                               (150, 0, 250): '乔木林地',
                               (150, 150, 250): '灌木林地',
                               (250, 200, 0): '天然草地',
                               (200, 200, 0): '人工草地',
                               (200, 0, 0): '工业用地',
                               (250, 0, 150): '城市住宅',
                               (200, 150, 150): '村镇住宅',
                               (250, 150, 150): '交通运输',
                               (0, 0, 200): '河流',
                               (0, 150, 200): '湖泊',
                               (0, 200, 250): '坑塘',
                               (0, 0, 0): '其他类别'} )

color_index_map = OrderedDict( {(0, 200, 0): 0,
                                (150, 250, 0): 1,
                                (150, 200, 150): 2,
                                (200, 0, 200): 3,
                                (150, 0, 250): 4,
                                (150, 150, 250): 5,
                                (250, 200, 0): 6,
                                (200, 200, 0): 7,
                                (200, 0, 0): 8,
                                (250, 0, 150): 9,
                                (200, 150, 150): 10,
                                (250, 150, 150): 11,
                                (0, 0, 200): 12,
                                (0, 150, 200): 13,
                                (0, 200, 250): 14,
                                (0, 0, 0): 15} )


def statistic_label(path, name):
    file_image = Image.open( os.path.join( path, name ) )
    color_map_number = OrderedDict( {key: 0 for key, _ in color_name_map.items()} )
    np_image = np.array( file_image )
    np_image_size = np_image.shape
    for h in range( np_image_size[0] ):
        for w in range( np_image_size[1] ):
            value = tuple( np_image[h, w, :].tolist() )
            color_map_number[value] += 1
    num_list = [name]
    for _, v in color_map_number.items():
        num_list.append( v )
    return num_list


def testOne():
    print( color_name_map )
    name_list = ["文件名"]
    for _, v in color_name_map.items():
        name_list.append( v )
    print( name_list )
    all_statistic_list = []

    num_list = statistic_label(
        '/home/arron/Documents/grey/Project_Rssrai/rssrai/split/label',
        'GF2_PMS1__20150212_L1A0000647768-MSS1_label_0_0.tif' )
    all_statistic_list.append( num_list )
    df = pd.DataFrame( all_statistic_list, columns=name_list )
    print( df )


def testSplitLabel():
    print( color_name_map )
    name_list = ["文件名"]
    for _, v in color_name_map.items():
        name_list.append( v )
    print( name_list )
    all_statistic_list = []
    base_path = Path.db_root_dir( "rssrai" )
    image_path = os.path.join( base_path, "split", "label" )
    from glob import glob
    image_list = glob( os.path.join( image_path, "*.tif" ) )
    # 多进程
    pool = Pool( 16 )
    for image in tqdm( image_list ):
        list = image.split( "/" )
        path = "/".join( list[:-1] )
        name = list[-1]

        result = pool.apply_async( statistic_label, args=(path, name) )
        all_statistic_list.append( result.get() )

    df = pd.DataFrame( all_statistic_list, columns=name_list )
    print( df )
    df.to_csv( os.path.join( base_path, "split_label.csv" ) )


def testTrainLabel():
    print( color_name_map )
    name_list = ["文件名"]
    for _, v in color_name_map.items():
        name_list.append( v )
    print( name_list )
    all_statistic_list = []
    base_path = Path.db_root_dir( "rssrai" )
    image_path = os.path.join( base_path, "train", "label" )
    from glob import glob
    image_list = glob( os.path.join( image_path, "*.tif" ) )
    # 多进程
    pool = Pool( 16 )
    for image in tqdm( image_list ):
        list = image.split( "/" )
        path = "/".join( list[:-1] )
        name = list[-1]

        result = pool.apply_async( statistic_label, args=(path, name) )
        all_statistic_list.append( result.get() )

    df = pd.DataFrame( all_statistic_list, columns=name_list )
    print( df )
    df.to_csv( os.path.join( base_path, "train_label.csv" ) )


def test_valid_label():
    print( color_name_map )
    name_list = ["文件名"]
    for _, v in color_name_map.items():
        name_list.append( v )
    print( name_list )
    all_statistic_list = []
    base_path = Path.db_root_dir( "rssrai" )
    image_path = os.path.join( base_path, "split_train", "label" )
    from glob import glob
    image_list = glob( os.path.join( image_path, "*.tif" ) )
    # 多进程
    pool = Pool( 16 )
    for image in tqdm( image_list ):
        list = image.split( "/" )
        path = "/".join( list[:-1] )
        name = list[-1]

        result = pool.apply_async( statistic_label, args=(path, name) )
        all_statistic_list.append( result.get() )

    df = pd.DataFrame( all_statistic_list, columns=name_list )
    print( df )
    df.to_csv( os.path.join( "valid_label.csv" ) )


if __name__ == '__main__':
    test_valid_label()
