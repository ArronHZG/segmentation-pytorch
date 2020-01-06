# cython: language_level=3
from .open.ade20k import ADE20KSegmentation
from .open.base import *
from .open.cityscapes import CitySegmentation
from .open.coco import COCOSegmentation
from .open.pascal_aug import VOCAugSegmentation
from .open.pascal_voc import VOCSegmentation
from .open.pcontext import ContextSegmentation
from .private.cloud.cloud import Cloud
from .private.rssrai.rssrai import Rssrai
from .private.xian.xian import Xian

datasets = {
    'coco': COCOSegmentation,
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'pcontext': ContextSegmentation,
    'citys': CitySegmentation,
    'rssrai': Rssrai,
    'xian': Xian,
    'cloud': Cloud
}


def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
