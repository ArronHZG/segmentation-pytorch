import os
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

color_name_map = OrderedDict({(0, 200, 0): '水田',
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
                              (0, 0, 0): '其他类别'})

color_index_map = OrderedDict({(0, 200, 0): 0,
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
                               (0, 0, 0): 15})

color_list = np.array([[0, 200, 0],
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

mean = (0.52891074, 0.38070734, 0.40119018, 0.36884733)
std = (0.24007008, 0.23784, 0.22267079, 0.21865861)


def decode_seg_map_sequence(label_masks):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


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
    for ll in range(0, len(color_list)):
        r[label_mask == ll] = color_list[ll, 0]
        g[label_mask == ll] = color_list[ll, 1]
        b[label_mask == ll] = color_list[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
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
    label_image = label_image.astype(int)
    label_mask = np.zeros((label_image.shape[0], label_image.shape[1]), dtype=np.int16)
    for ii, label in enumerate(color_list):
        label_mask[np.where(np.all(label_image == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def visualize_batch_image(image, target, output,
                          epoch, batch_index,
                          directory):
    # image (B,C,H,W) To (B,H,W,C)
    image_np = image.cpu().numpy()
    image_np = np.transpose(image_np, axes=[0, 2, 3, 1])
    image_np *= std
    image_np += mean
    image_np *= 255.0
    image_np = image_np.astype(np.uint8)

    # target (B,H,W)
    target = target.cpu().numpy()

    # output (B,C,H,W) to (B,H,W)
    output = torch.argmax(output, dim=1).cpu().numpy()
    # print(output)
    # print(output.shape)

    # # blank (H,W,C)
    # blank = np.zeros((output.shape[1],output.shape[2],3))
    # blank.fill(255)

    for i in range(min(3, image_np.shape[0])):
        img_tmp = image_np[i]
        img_rgb_tmp = np.array(Image.fromarray(img_tmp).convert("RGB")).astype(np.uint8)
        target_rgb_tmp = decode_segmap(target[i]).astype(np.uint8)
        output_rgb_tmp = decode_segmap(output[i]).astype(np.uint8)
        plt.figure()
        plt.title('display')
        plt.subplot(131)
        plt.imshow(img_rgb_tmp, vmin=0, vmax=255)
        plt.subplot(132)
        plt.imshow(target_rgb_tmp, vmin=0, vmax=255)
        plt.subplot(133)
        plt.imshow(output_rgb_tmp, vmin=0, vmax=255)

        path = os.path.join(directory, "train_image", f'epoch_{epoch}')
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f"{path}/{batch_index}-{i}.jpg")
        plt.close('all')
