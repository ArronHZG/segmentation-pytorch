import numpy as np

from experiments.datasets import VOCSegmentation, Rssrai
from experiments.datasets.private.xian import xian


def make_data_loader(dataset_name, base_size, crop_size, basic_dir):
    if dataset_name == 'xian':
        train_set = xian.Xian(mode='train')
        val_set = xian.Xian(mode='valid')
        num_class = train_set.NUM_CLASSES

        return train_set, val_set, num_class

    if dataset_name == 'rssrai':
        train_set = Rssrai(mode='train',
                           base_size=base_size,
                           crop_size=crop_size,
                           basic_dir=basic_dir,
                           is_load_numpy=True)
        val_set = Rssrai(mode='val',
                         base_size=base_size,
                         crop_size=crop_size,
                         basic_dir=basic_dir,
                         is_load_numpy=True)
        num_class = train_set.NUM_CLASSES

        return train_set, val_set, num_class

    if dataset_name == 'pascal_voc':
        train_set = VOCSegmentation(split='train', mode='train', base_size=base_size, crop_size=crop_size)
        val_set = VOCSegmentation(split='val', mode='val', base_size=base_size, crop_size=crop_size)
        num_class = train_set.NUM_CLASSES

        return train_set, val_set, num_class

    else:
        raise NotImplementedError


def get_labels(label_number):
    """
    :return: (19 , 3)
    """
    label_19 = np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])

    label_21 = np.array([
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128]])

    label_16 = np.array([[0, 200, 0],
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
                         [0, 0, 0]])

    label_colors = {19: label_19, 21: label_21, 16: label_16}
    return label_colors[label_number]


def decode_segmap(label_mask, label_number):
    """Decode segmentation class labels into a color image
        :param label_mask:
        :param label_number:
    """
    color_list = get_labels(label_number)
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, len(color_list)):
        r[label_mask == ll] = color_list[ll, 0]
        g[label_mask == ll] = color_list[ll, 1]
        b[label_mask == ll] = color_list[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb.astype(np.uint8)
