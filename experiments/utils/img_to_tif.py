import gc
import glob
import json
import os
import os.path as osp
import sys
import time
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from osgeo import gdal
from osgeo import ogr
from tqdm import tqdm


class TifImage:
    def __init__(self, xOffset, yOffset, xScale, yScale, image_width, image_height, image):
        self.xOffset = xOffset
        self.yOffset = yOffset
        self.xScale = xScale
        self.yScale = yScale
        self.image_width = image_width
        self.image_height = image_height
        self.image = image


def parseIMG(imgPath, channel=None, channel_list=None):
    if channel is None:
        channel_list = [1, 2, 3, 4]
        channel = len(channel_list)

    ds = gdal.Open(imgPath)
    print("\t+++ RasterCount:", ds.RasterCount)
    geo_transform = ds.GetGeoTransform()

    xOffset = geo_transform[0]
    yOffset = geo_transform[3]
    xScale = geo_transform[1]
    yScale = geo_transform[5]

    image_width = ds.RasterXSize
    image_height = ds.RasterYSize

    print("\t+++ X offset:", xOffset)
    print("\t+++ Y offset:", yOffset)
    print("\t+++ X scale:", xScale)
    print("\t+++ Y scale:", yScale)
    print("\t+++ Image height:", image_height)
    print("\t+++ Image width:", image_width)

    bands = []
    if channel > ds.RasterCount:
        raise RuntimeError("channel > ds.RasterCount")
    bands.append(ds.GetRasterBand(channel_list[0]).ReadAsArray())
    bands.append(ds.GetRasterBand(channel_list[1]).ReadAsArray())
    bands.append(ds.GetRasterBand(channel_list[2]).ReadAsArray())
    bands.append(ds.GetRasterBand(channel_list[3]).ReadAsArray())
    image = np.stack(bands, axis=-1).astype(np.uint8)

    return TifImage(xOffset, yOffset, xScale, yScale, image_width, image_height, image)
