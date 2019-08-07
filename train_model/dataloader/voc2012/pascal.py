import cv2
import os

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
import albumentations as A
from albumentations.pytorch import ToTensor
import numpy as np

from train_model.config.mypath import Path

pascal_labels = np.asarray( [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                             [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                             [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                             [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                             [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                             [0, 64, 128]] )


class VOCSegmentation( Dataset ):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 21

    def __init__(self,
                 type,
                 base_size=(513, 513),
                 crop_size=(256, 256),
                 base_dir=Path.db_root_dir( 'pascal' )
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param type: train/val
        :param transform: transform to apply
        """
        assert type in ['train', 'val']
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join( self._base_dir, 'JPEGImages' )
        self._cat_dir = os.path.join( self._base_dir, 'SegmentationClass' )

        self.split = type
        self.base_size = base_size
        self.crop_size = crop_size
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.im_ids = []
        self.images = []
        self.categories = []

        _splits_dir = os.path.join( self._base_dir, 'ImageSets', 'Segmentation' )
        with open( os.path.join( os.path.join( _splits_dir, self.split + '.txt' ) ), "r" ) as f:
            lines = f.read().splitlines()
        for ii, line in enumerate( lines ):
            _image = os.path.join( self._image_dir, line + ".jpg" )
            _cat = os.path.join( self._cat_dir, line + ".png" )
            assert os.path.isfile( _image )
            assert os.path.isfile( _cat )
            self.im_ids.append( line )
            self.images.append( _image )
            self.categories.append( _cat )

        assert (len( self.images ) == len( self.categories ))

        # Display stats
        print( 'Number of images in {}: {:d}'.format( type, len( self.images ) ) )

    def __len__(self):
        # return 5
        return len( self.images )

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair( index )
        sample = {'image': _img, 'label': _target}

        if self.split == "train":
            return self.transform_tr( sample )
        elif self.split == 'val':
            return self.transform_val( sample )

    def _make_img_gt_point_pair(self, index):
        _img = np.array( Image.open( self.images[index] ).convert( 'RGB' ) )
        _target = np.array( Image.open( self.categories[index] ) )

        return _img, _target

    def transform_tr(self, sample):

        compose = A.Compose( [
            A.PadIfNeeded( self.base_size[0], self.base_size[1], p=1 ),
            A.RandomCrop( self.crop_size[0], self.crop_size[1], p=1 ),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RGBShift(),
            A.Blur(),
            A.GaussNoise(),
            A.Normalize( mean=self.mean, std=self.std, p=1 ),
        ], additional_targets={'image': 'image', 'label': 'mask'} )

        return self.toTorchTensor( compose( **sample ) )

    def transform_val(self, sample):

        compose = A.Compose( [
            A.PadIfNeeded( self.base_size[0], self.base_size[1], p=1 ),
            A.CenterCrop( self.crop_size[0], self.crop_size[1], p=1 ),
            A.Normalize( mean=self.mean, std=self.std, p=1 ),
        ], additional_targets={'image': 'image', 'label': 'mask'} )

        return self.toTorchTensor( compose( **sample ) )

    def toTorchTensor(self, sample):
        sample['image'] = torch.from_numpy( sample['image'] ).permute( 2, 0, 1 )
        sample['label'] = torch.from_numpy( sample['label'] ).long()
        return sample

    def __str__(self):
        return 'VOC2012(split=' + str( self.split ) + ')'

    def decode_seg_map_sequence(self, label_masks):
        rgb_masks = []
        for label_mask in label_masks:
            rgb_mask = self.decode_segmap( label_mask )
            rgb_masks.append( rgb_mask )
        rgb_masks = torch.from_numpy( np.array( rgb_masks ).transpose( [0, 3, 1, 2] ) )
        return rgb_masks

    def decode_segmap(self, label_mask):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """

        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range( 0, self.NUM_CLASSES ):
            r[label_mask == ll] = pascal_labels[ll, 0]
            g[label_mask == ll] = pascal_labels[ll, 1]
            b[label_mask == ll] = pascal_labels[ll, 2]
        rgb = np.zeros( (label_mask.shape[0], label_mask.shape[1], 3) )
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype( int )
        label_mask = np.zeros( (mask.shape[0], mask.shape[1]), dtype=np.int16 )
        for ii, label in enumerate( pascal_labels ):
            label_mask[np.where( np.all( mask == label, axis=-1 ) )[:2]] = ii
        label_mask = label_mask.astype( int )
        return label_mask


def show_image():
    from torch.utils.data import DataLoader, Dataset
    import matplotlib.pyplot as plt

    base_size = (513, 513)
    crop_size = (512, 512)

    voc = VOCSegmentation( type='val',
                           base_size=base_size, crop_size=crop_size )

    data_loader = DataLoader( voc, batch_size=5, shuffle=True, num_workers=8 )

    for ii, sample in enumerate( data_loader ):
        for jj in range( sample["image"].size()[0] ):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            print( img.shape )
            print( gt.shape )
            tmp = np.array( gt[jj] ).astype( np.uint8 )
            segmap = voc.decode_segmap( tmp )
            img_tmp = np.transpose( img[jj], axes=[1, 2, 0] )
            img_tmp *= voc.std
            img_tmp += voc.mean
            img_tmp *= 255.0
            img_tmp = img_tmp.astype( np.uint8 )
            plt.figure()
            plt.title( 'display' )
            plt.subplot( 211 )
            plt.imshow( img_tmp )
            plt.subplot( 212 )
            plt.imshow( segmap )

        if ii == 1:
            break

    plt.show( block=True )
if __name__ == '__main__':
    image = Image.open( "/home/arron/Documents/arron/d2l-zh/data/VOCdevkit/VOC2012/SegmentationClass/2008_005231.png" )
    _target = np.array(image )
    # image = Image.fromarray( _target.astype( 'uint8' ) )
    # image.mode = "RGB"
    # image.save( "./a.png" )
