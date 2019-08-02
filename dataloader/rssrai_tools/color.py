from collections import OrderedDict
from pprint import pprint

import numpy as np

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

color_list = np.array( [[0, 200, 0],
                        [150, 250, 0],
                        [150, 200, 150],
                        [200, 0, 200],
                        [150, 0, 250],
                        [150, 150, 250],
                        [250, 200, 0],
                        [200, 200, 0],
                        [200, 0, 0],
                        [250, 0, 150],
                        [200, 150, 150],
                        [250, 150, 150],
                        [0, 0, 200],
                        [0, 150, 200],
                        [0, 200, 250],
                        [0, 0, 0]] )


def decode_segmap(label_mask):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range( 0, len( color_list ) ):
        r[label_mask == ll] = color_list[ll, 0]
        g[label_mask == ll] = color_list[ll, 1]
        b[label_mask == ll] = color_list[ll, 2]
    rgb = np.zeros( (label_mask.shape[0], label_mask.shape[1], 3) )
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


def encode_segmap(label_image):
    """Encode segmentation label images as pascal classes
    Args:
        label_image (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    label_image = label_image.astype( int )
    label_mask = np.zeros( (label_image.shape[0], label_image.shape[1]), dtype=np.int16 )
    for ii, label in enumerate(color_list):
        label_mask[np.where( np.all( label_image == label, axis=-1 ) )[:2]] = ii
    label_mask = label_mask.astype( int )
    return label_mask


def test_encode():
    from PIL import Image
    image = Image.open('/home/arron/Documents/grey/Project_Rssrai/rssrai/split_valid/label/GF2_PMS1__20150212_L1A0000647768-MSS1_label_0_0_0_0.tif')
    image = np.array(image)
    mask = encode_segmap(image)
    for i in range(image.shape[1]):
        pprint(image[0,i])
        pprint(mask[0,i])

if __name__ == '__main__':
    test_encode()