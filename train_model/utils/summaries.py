import os
import torch
import numpy as np
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from train_model.dataloader.rssrai_tools.rssrai import Rssrai


class TensorboardSummary():
    def __init__(self, directory, dataset):
        self.directory = directory
        self.writer = SummaryWriter( logdir=os.path.join( self.directory ) )
        self.dataset = Rssrai()
        plt.axis( 'off' )

    def visualize_image(self, image, target, output, global_step):
        # image (B,C,H,W) to (B,H,W,C)
        image = image.cpu().numpy()
        image = image[:, 1:, :, :]

        # target (B,H,W)
        target = target.cpu().numpy()

        # output (B,C,H,W) to (B,H,W)
        output = torch.argmax( output, dim=1 ).cpu().numpy()

        for i in range( min( 10, image.shape[0] ) ):
            img_tmp = np.transpose( image[i], axes=[1, 2, 0] )
            img_tmp *= self.dataset.std[1:]
            img_tmp += self.dataset.mean[1:]
            img_tmp *= 255.0
            img_tmp = img_tmp.astype( np.uint8 )
            target_rgb_tmp = self.dataset.decode_segmap( target[i] )
            output_rgb_tmp = self.dataset.decode_segmap( output[i] )
            plt.figure()
            plt.title( 'display' )
            plt.subplot( 131 )
            plt.imshow( img_tmp )
            plt.subplot( 132 )
            plt.imshow( target_rgb_tmp )
            plt.subplot( 133 )
            plt.imshow( output_rgb_tmp )

            path = f'{self.directory}/epoch_{global_step}'
            if not os.path.exists( path ):
                os.makedirs( path )
            plt.savefig( f"{path}/{i}.jpg" )

            plt.close('all')

        # self.grid_image = make_grid( image[:3,1:].clone().cpu().data, 3, normalize=True )
        # self.writer.add_image( 'Image', grid_image, global_step )
        # self.grid_image = make_grid( decode_seg_map_sequence( torch.max( output[:3], 1 )[1].detach().cpu().numpy()), 3, normalize=False, range=(0, 255) )
        # self.writer.add_image( 'Predicted label', grid_image, global_step )
        # self.grid_image = make_grid( decode_seg_map_sequence( torch.squeeze( target[:3], 1 ).detach().cpu().numpy()), 3, normalize=False, range=(0, 255) )
        # self.writer.add_image( np.array(plt), global_step )
