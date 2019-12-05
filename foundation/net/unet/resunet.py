import torch.nn as nn

from .model import getBackBone
from foundation.net.unet.model.layers import unetUp
from .utils import init_weights


class ResUNet(nn.Module):

    def __init__(self, num_classes, in_channels, backbone, pretrained, is_deconv=True, is_batchnorm=True):
        super(ResUNet, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.is_deconv = is_deconv
        self.backbone = getBackBone(backbone, in_channels, pretrained)
        self.is_batchnorm = is_batchnorm

        filters = [128, 256, 512, 1024, 2048]

        for filter in filters:
            one_conv2 = nn.Conv2d(filter, int(filter / 8), 1, 1)
            setattr(self, f'down_channel_{filter}', one_conv2)

        filters = [int(x / 8) for x in filters]
        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], num_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):

        # input   [2,   3, 512, 512]
        # output0 [2, 128, 512, 512]
        # output1 [2, 256, 256, 256]
        # output2 [2, 512, 128, 128]
        # output3 [2, 1024, 64, 64]
        # output4 [2, 2048, 32, 32]

        output0, output1, output2, output3, output4 = self.backbone(inputs)
        output0 = self.down_channel_128(output0)
        output1 = self.down_channel_256(output1)
        output2 = self.down_channel_512(output2)
        output3 = self.down_channel_1024(output3)
        output4 = self.down_channel_2048(output4)

        up4 = self.up_concat4(output4, output3)  # 128*64*64
        up3 = self.up_concat3(up4, output2)  # 64*128*128
        up2 = self.up_concat2(up3, output1)  # 32*256*256
        up1 = self.up_concat1(up2, output0)  # 16*512*512
        final = self.final(up1)

        return final
