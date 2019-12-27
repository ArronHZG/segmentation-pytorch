import glob
import os
import time
import sys

import cv2
import gdal
import numpy as np
import shapefile
from PIL import Image
from imantics import Mask

from experiments.utils.img_to_tif import TifImage

np.set_printoptions(threshold=sys.maxsize)
Image.MAX_IMAGE_PIXELS = None

END_TIME = time.strptime("2020 3 25 20 09", "%Y %m %d %H %M")

color_list = []
for r in [255, 120, 180, 250]:
    for g in [255, 120, 180, 250]:
        for b in [255, 120, 180, 250]:
            color_list.append([r, g, b])


def check_time():
    cur_time = time.localtime(time.time())

    e_time = int(time.mktime(END_TIME))
    c_time = int(time.mktime(cur_time))

    if e_time > c_time:
        return True
    return False


def func_time(func):
    def inner(*args, **kw):
        start_time = time.time()
        func(*args, **kw)
        end_time = time.time()
        print('run time：', end_time - start_time, 's')

    return inner


def numpy_to_shp(n_class, tif_image: TifImage, inNumpyPath, outShpPath, code):
    if not os.path.exists(outShpPath):
        os.makedirs(outShpPath)

    xo = tif_image.xOffset
    yo = tif_image.yOffset
    xs = tif_image.xScale
    ys = tif_image.yScale

    whole_result = np.load(os.path.join(inNumpyPath, code + ".npz"))["pred"].astype(np.uint8)
    print(f'\t{whole_result.shape}')
    w = shapefile.Writer(os.path.join(outShpPath, code + ".shp"))
    w.field("DLBM", "C")
    w.field("DLMC", "C")
    for i in range(1, n_class):
        print(f"{i}/{n_class - 1}")
        if np.any(whole_result == i):
            mask = (whole_result == i).astype(np.uint8)
            polygons = Mask(mask).polygons()
            for pp in polygons.points:
                pp = pp.astype(np.float32)
                pp[:, 0] *= xs
                pp[:, 1] *= ys
                pp[:, 0] += xo
                pp[:, 1] += yo
                if cv2.contourArea(pp) < 1000:
                    continue
                w.poly([pp[::-1].tolist()])
                w.record(i, i)
    w.close()


def decode_segmap(n_class, label_mask):
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(1, n_class):
        r[label_mask == ll] = color_list[ll - 1][0]
        g[label_mask == ll] = color_list[ll - 1][1]
        b[label_mask == ll] = color_list[ll - 1][2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb


def numpy_to_vis(n_class, inNumpyPath, visPath, code, times=8):
    if not os.path.exists(visPath):
        os.makedirs(visPath)
    whole_result = np.load(os.path.join(inNumpyPath, code + ".npz"))["pred"].astype(np.uint8)
    rgb = decode_segmap(n_class, whole_result)
    image = Image.fromarray(rgb.astype('uint8'))
    if times is not 1:
        h = rgb.shape[0]
        w = rgb.shape[1]
        image = image.resize((w // times, h // times), Image.ANTIALIAS)
    image.save(os.path.join(visPath, code + ".png"))


def test():
    inImgPath = '/mnt/anna-fuse/taihe/fexion/yaogan/ruiyan_debug/source_data/img'
    outShpPath = '/mnt/anna-fuse/taihe/fexion/yaogan/ruiyan_debug/pre_data/shp'
    inNumpyPath = '/home/arron/PycharmProjects/segmentation-pytorch/script/run/cloud/fcn-resnet50/experiment_4/test_output'
    visPath = '/home/arron/PycharmProjects/segmentation-pytorch/script/run/cloud/fcn-resnet50/experiment_4/vis'
    n_class = 2

    code_list = ["img"]
    for code in code_list:
        numpy_to_shp(n_class, inImgPath, inNumpyPath, outShpPath, code)
        # numpy_to_vis(n_class, inNumpyPath, visPath, code)


if __name__ == '__main__':
    test()
