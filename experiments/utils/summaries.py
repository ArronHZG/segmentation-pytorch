# cython: language_level=3
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from tqdm import tqdm


def save_image(np_image, path, name, mode):
    image = Image.fromarray(np_image.astype('uint8'))
    image.mode = mode
    image.save(os.path.join(path, name))


# def merge_rssrai_test_label_images(image_path, save_image_path):
#     import pandas as pd
#     df = pd.read_csv("test_name_list.csv")
#     name_list = df['name'].tolist()
#     for name in tqdm(name_list):
#         merge_image(image_path,
#                     name[:-4] + "_label.tif",
#                     save_image_path,
#                     "RGB")


def save_test_image(self, name_list, output):
    # output (B,C,H,W) to (B,H,W)
    # print(output.size())
    output = torch.argmax(output, dim=1).cpu().numpy()
    # print(output.shape)
    # print(output)
    for i, name in enumerate(name_list):
        output_rgb_tmp = self.dataset.decode_segmap(output[i])
        # print(output_rgb_tmp)
        name = f'{name[:-4]}_label.tif'
        # print(name)
        save_image(output_rgb_tmp, self.output_label_path, name, "RGB")

    # mean = (0.52891074, 0.38070734, 0.40119018, 0.36884733)
    # std = (0.24007008, 0.23784, 0.22267079, 0.21865861)
    #
    # # image (B,C,H,W) To (B,H,W,C)
    # image_np = output.cpu().numpy()
    # image_np = np.transpose(image_np, axes=[0, 2, 3, 1])
    # image_np *= std
    # image_np += mean
    # image_np *= 255.0
    # image_np = image_np.astype(np.uint8)


class TensorboardSummary:
    def __init__(self, directory):
        self.directory = directory
        self.writer = SummaryWriter(self.directory)

        # self.output_label_path = os.path.join(self.directory, "test_output")
        # if not os.path.exists(self.output_label_path):
        #     os.makedirs(self.output_label_path)
        #
        # self.merge_path = os.path.join(self.directory, "test_rgb_label")
        # if not os.path.exists(self.merge_path):
        #     os.makedirs(self.merge_path)
        plt.axis('off')
