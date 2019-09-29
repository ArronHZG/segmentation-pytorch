from .open import VOCSegmentation
from .private.rssrai_tools import rssrai
from .private.xian import xian


def make_data_loader(dataset_name, base_size, crop_size):
    if dataset_name == 'xian':
        train_set = xian.Xian(mode='train')
        val_set = xian.Xian(mode='valid')
        num_class = train_set.NUM_CLASSES

        return train_set, val_set, num_class

    if dataset_name == 'rssrai':
        train_set = rssrai.Rssrai(mode='train', base_size=base_size, crop_size=crop_size)
        val_set = rssrai.Rssrai(mode='valid', base_size=base_size, crop_size=crop_size)
        num_class = train_set.NUM_CLASSES

        return train_set, val_set, num_class

    if dataset_name == 'pascal_voc':
        train_set = VOCSegmentation(split='train', mode='train', base_size=base_size, crop_size=crop_size)
        val_set = VOCSegmentation(split='val', mode='val', base_size=base_size, crop_size=crop_size)
        num_class = train_set.NUM_CLASSES

        return train_set, val_set, num_class

    else:
        raise NotImplementedError
