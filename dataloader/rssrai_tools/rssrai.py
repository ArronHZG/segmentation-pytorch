import functools
import os
import random
from pprint import pprint

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot  as plt

from dataloader.rssrai_tools.color import encode_segmap, decode_segmap
from utils.mypath import Path


class Rssrai( data.Dataset ):
    NUM_CLASSES = 16

    def __init__(self,
                 base_dir=Path.db_root_dir( 'rssrai' ),
                 type='train'):

        super().__init__()
        self._base_dir = base_dir
        self.type = type
        self.mean = (0.52891074, 0.38070734, 0.40119018, 0.36884733)
        self.std = (0.24007008, 0.23784, 0.22267079, 0.21865861)
        self.im_ids = []
        self.images = []
        self.categories = []

        # 加载数据
        if self.type == 'train':
            train_csv = os.path.join( self._base_dir, 'train_name.csv' )
            self._label_name_list = pd.read_csv( train_csv )["name"].values.tolist()

            self._image_dir = os.path.join( self._base_dir, 'split_train', 'img' )
            self._label_dir = os.path.join( self._base_dir, 'split_train', 'label' )

            self.len = 30000

        if self.type == 'valid':
            valid_csv = os.path.join( self._base_dir, 'valid_name.csv' )
            self._label_name_list = pd.read_csv( valid_csv )["name"].values.tolist()
            self._image_dir = os.path.join( self._base_dir, 'split_valid', 'img' )
            self._label_dir = os.path.join( self._base_dir, 'split_valid', 'label' )

            self.len = len( self._label_name_list )

        if self.type == 'test':
            pass

    def __getitem__(self, index):
        return self.transform( self.get_numpy_image( index ) )

    def __len__(self):
        return self.len

    def get_numpy_image(self, index):
        '''
        训练集随机选一张图片,然后随机crop
        验证集按顺序选取
        测试集按顺序选取
        '''
        if self.type == 'train':
            name = self._get_random_file_name()
            sample = self._read_file( name )
            sample = self._random_crop( sample )
            return sample
        if self.type == 'valid':
            sample = self._read_file( self._label_name_list[index] )
            sample = self._random_crop( sample )
            return sample
        if self.type == 'test':
            pass
        return 1, 2

    def _random_crop(self, sample):
        aug = A.RandomCrop( 256, 256, p=1 )
        compose = A.Compose( [aug], additional_targets={'image': 'image', 'label': 'mask'} )
        return compose( **sample )

    @functools.lru_cache( maxsize=None )
    def _read_file(self, label_name):
        image_name = label_name.replace( "_label", "" )
        image_pil = Image.open( os.path.join( self._image_dir, image_name ) )
        image_np = np.array( image_pil )

        label_pil = Image.open( os.path.join( self._label_dir, label_name ) )
        label_np = np.array( label_pil )
        label_mask = encode_segmap( label_np )

        return {'image': image_np, 'label': label_mask}

    def _get_random_file_name(self):
        return random.choice( self._label_name_list )

    def transform(self, sample):
        sample['image'] = A.Normalize( mean=self.mean, std=self.std )( image=sample['image'] )["image"]
        sample['image'] = torch.from_numpy( sample['image'] ).permute( 2, 0, 1 )
        sample['label'] = torch.from_numpy( sample['label'] )
        return sample

    def __str__(self):
        return 'Rssrai(split=' + str( self.split ) + ')'


def testData():
    test_path = os.path.join( Path().db_root_dir( "rssrai" ), "测试输出" )
    if not os.path.exists( test_path ):
        os.makedirs( test_path )

    rssrai = Rssrai(type="valid")
    for i in rssrai:
        pprint( i["image"].shape )
        pprint( i["label"].shape )
        break
    data_loader = DataLoader( rssrai, batch_size=4, shuffle=True, num_workers=0 )

    for ii, sample in enumerate( data_loader ):
        sample['image'] = sample['image'][:, 1:, :, :]
        for jj in range( sample["image"].size()[0] ):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            img_tmp = np.transpose( img[jj], axes=[1, 2, 0] )
            tmp = gt[jj]
            segmap = decode_segmap( tmp )
            img_tmp *= rssrai.std[1:]
            img_tmp += rssrai.mean[1:]
            img_tmp *= 255.0
            img_tmp = img_tmp.astype( np.uint8 )
            plt.figure()
            plt.title( 'display' )
            plt.subplot( 121 )
            plt.imshow( img_tmp )
            plt.subplot( 122 )
            plt.imshow( segmap )
            # with open( f"{test_path}/rssrai-{ii}-{jj}.txt", "w" ) as f:
            #     f.write( str( img_tmp ) )
            #     f.write( str( tmp ) )
            #     f.write( str( segmap ) )
            plt.savefig( f"{test_path}/rssrai-{ii}-{jj}.jpg" )

        if ii == 5:
            break

    plt.show( block=True )


if __name__ == '__main__':
    testData()
