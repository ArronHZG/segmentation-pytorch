import torch
import torch.nn as nn

from .models import getBackBone


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding

        for i in range(1, n + 1):
            conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                 nn.BatchNorm2d(out_size),
                                 nn.ReLU(inplace=True), )
            setattr(self, 'conv%d' % i, conv)
            in_size = out_size

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x


#
# class unetConv2(nn.Sequential):
#     def __init__(self, in_size, out_size, n=2, kernel_size=3, stride=1, padding=1):
#         layers = [
#             nn.ConvTranspose2d(num_classes, num_classes, kernel_size,
#                                stride=stride, padding=padding, bias=False)
#         ]
#         super(FCNUpsampling, self).__init__(*layers)


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_size, out_size, 1))

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)


class UNet(nn.Module):

    def __init__(self, num_classes, in_channels, backbone, pretrained):
        super(UNet, self).__init__()
        self.in_channels = in_channels

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling

        self.backbone = getBackBone(backbone, in_channels, pretrained)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights

    def forward(self, inputs):

        # pool1  scaling = 1/4   channel = 256
        # pool2  scaling = 1/8   channel = 512
        # pool3  scaling = 1/16  channel = 1024
        # pool4  scaling = 1/32  channel = 2048
        pool1, pool2, pool3, pool4 = self.backbone( x )

        conv1 = self.conv1(inputs)  # 16*512*512
        maxpool1 = self.maxpool(conv1)  # 16*256*256

        conv2 = self.conv2(maxpool1)  # 32*256*256
        maxpool2 = self.maxpool(conv2)  # 32*128*128

        conv3 = self.conv3(maxpool2)  # 64*128*128
        maxpool3 = self.maxpool(conv3)  # 64*64*64

        conv4 = self.conv4(maxpool3)  # 128*64*64
        maxpool4 = self.maxpool(conv4)  # 128*32*32

        center = self.center(maxpool4)  # 256*32*32
        up4 = self.up_concat4(center, conv4)  # 128*64*64
        up3 = self.up_concat3(up4, conv3)  # 64*128*128
        up2 = self.up_concat2(up3, conv2)  # 32*256*256
        up1 = self.up_concat1(up2, conv1)  # 16*512*512

        final = self.final(up1)

        return final
