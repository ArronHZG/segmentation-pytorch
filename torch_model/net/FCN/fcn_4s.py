from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from models import getBackBone


class FCNHead(nn.Sequential):
    '''
    To merge the feature mapping with different scale in the middle with the feature mapping by
       upsampling need to change channel dimensionality to the same.
    '''

    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)


class FCNUpsampling(nn.Sequential):
    '''

    '''

    def __init__(self, num_classes, kernel_size, stride=1, padding=0):
        layers = [
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size,
                               stride=stride, padding=padding, bias=False)
        ]
        super(FCNUpsampling, self).__init__(*layers)


class FCN(nn.Module):
    def __init__(self, backbone, num_classes, in_channels, pretrained, aux_classifier=None):
        super(FCN, self).__init__()
        # Using the modified resNet to get 4 different scales of the tensor,
        # in fact, the last three used in the paper,
        # first reserved for experiment
        self.backbone = getBackBone(backbone, in_channels, pretrained, )
        self.pool1_FCNHead = FCNHead(256, num_classes)
        self.pool2_FCNHead = FCNHead(512, num_classes)
        self.pool3_FCNHead = FCNHead(1024, num_classes)
        self.pool4_FCNHead = FCNHead(2048, num_classes)

        # upsampling using transposeConvolution
        # out = s(in-1)+d(k-1)+1-2p
        # while s = s , d =1, k=2s, p = s/2, we will get out = s*in
        # we need to zoom in 32 times by 2 x 2 x 2 x 4
        self.up_score2 = FCNUpsampling(num_classes, 4, stride=2, padding=1)
        self.up_score4 = FCNUpsampling(num_classes, 8, stride=4, padding=2)
        self.up_score8 = FCNUpsampling(num_classes, 16, stride=8, padding=4)
        self.up_score32 = FCNUpsampling(num_classes, 64, stride=32, padding=16)

        self.aux_classifier = aux_classifier

        self.initial_weight()

    def forward(self, x):
        result = OrderedDict()

        input_shape = x.shape[-2:]
        # pool1  scaling = 1/4   channel = 256
        # pool2  scaling = 1/8   channel = 512
        # pool3  scaling = 1/16  channel = 1024
        # pool4  scaling = 1/32  channel = 2048
        pool1, pool2, pool3, pool4 = self.backbone(x)

        # pool1_same_channel  scaling = 1/4   channel = num_classes
        # pool2_same_channel  scaling = 1/8   channel = num_classes
        # pool3_same_channel  scaling = 1/16  channel = num_classes
        # pool4_same_channel  scaling = 1/32  channel = num_classes
        pool1_same_channel = self.pool1_FCNHead(pool1)
        pool2_same_channel = self.pool2_FCNHead(pool2)
        pool3_same_channel = self.pool3_FCNHead(pool3)
        pool4_same_channel = self.pool4_FCNHead(pool4)

        if self.aux_classifier is not None:
            result["aux"] = self.up_score32(pool4_same_channel)

        # merge x and pool3   scaling = 1/16
        x = self.up_score2(pool4_same_channel) + pool3_same_channel

        # merge x and pool2  scaling = 1/8
        x = self.up_score2(x) + pool2_same_channel

        # merge x and pool2  scaling = 1/4
        x = self.up_score2(x) + pool1_same_channel

        # scaling = 1
        result["out"] = self.up_score4(x)

        return result

    def initial_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight = torch.nn.Parameter(self.bilinear_kernel(m.in_channels, m.out_channels, m.kernel_size[0]))

    def bilinear_kernel(self, in_channels, out_channels, kernel_size):
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * \
               (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
        weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight)


if __name__ == '__main__':
    x = torch.rand((1, 3, 256, 256))

    m = getBackBone("fcnResNet50", 3, False)
    print(m)
    print(x.shape)
    a, b, c, d = m(x)
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape)

    m = FCN("fcnResNet50", num_classes=21, in_channels=3, pretrained=False)
    x = m(x)
    print(x['out'].size())

# show_model(m,"fcnResNet50")
