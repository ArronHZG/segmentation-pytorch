import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tensorboardX import SummaryWriter


class TensorboardSummary():
    def __init__(self, directory, dataset):
        self.directory = directory
        self.writer = SummaryWriter(logdir=os.path.join(self.directory))
        self.dataset = dataset
        plt.axis('off')

    def visualize_image(self, image, target, output, epoch, batch_index):
        # image (B,C,H,W) To (B,H,W,C)
        image_np = image.cpu().numpy()
        image_np = np.transpose(image_np, axes=[0,2,3,1])
        image_np *= self.dataset.std
        image_np += self.dataset.mean
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
            img_tmp=image_np[i]
            img_rgb_tmp = np.array(Image.fromarray(img_tmp).convert("RGB"))
            target_rgb_tmp = self.dataset.decode_segmap(target[i])
            output_rgb_tmp = self.dataset.decode_segmap(output[i])
            plt.figure()
            plt.title('display')
            plt.subplot(131)
            plt.imshow(img_rgb_tmp)
            plt.subplot(132)
            plt.imshow(target_rgb_tmp)
            plt.subplot(133)
            plt.imshow(output_rgb_tmp)

            path = f'{self.directory}/epoch_{epoch}'
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(f"{path}/{batch_index}-{i}.jpg")

            plt.close('all')

        # self.grid_image = make_grid( image[:3,1:].clone().cpu().data, 3, normalize=True )
        # self.writer.add_image( 'Image', grid_image, global_step )
        # self.grid_image = make_grid( decode_seg_map_sequence( torch.max( output[:3], 1 )[1].detach().cpu().numpy()), 3, normalize=False, range=(0, 255) )
        # self.writer.add_image( 'Predicted label', grid_image, global_step )
        # self.grid_image = make_grid( decode_seg_map_sequence( torch.squeeze( target[:3], 1 ).detach().cpu().numpy()), 3, normalize=False, range=(0, 255) )
        #     print(img_rgb_tmp.shape)
        #     print(target_rgb_tmp.shape)
        #     print(output_rgb_tmp.shape)
        #     print(blank.shape)
        #     cat_image = np.concatenate((img_rgb_tmp,
        #                                       blank,
        #                                       target_rgb_tmp,
        #                                       blank,
        #                                       output_rgb_tmp),axis=1)
        #     # print(cat_image.shape)
        #     self.writer.add_image('Image',
        #                       cat_image,
        #                           dataformats = "HWC"
        #                       )
