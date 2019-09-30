from .open.base import *
from .open.coco import COCOSegmentation
from .open.ade20k import ADE20KSegmentation
from .open.pascal_voc import VOCSegmentation
from .open.pascal_aug import VOCAugSegmentation
from .open.pcontext import ContextSegmentation
from .open.cityscapes import CitySegmentation
from .private.rssrai_tools.rssrai import Rssrai


datasets = {
    'coco': COCOSegmentation,
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'pcontext': ContextSegmentation,
    'citys': CitySegmentation,
    'rssrai': Rssrai,
}


def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
