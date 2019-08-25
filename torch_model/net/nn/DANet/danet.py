import torch
import torch.nn as nn

from .nn.attention import PAM_Module, CAM_Module


def conv2d(in_channels, out_channels, norm_layer):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                         norm_layer(out_channels),
                         nn.ReLU(inplace=True))


def dropout_conv1x1(in_channels, out_channels):
    return nn.Sequential(nn.Dropout2d(0.1, False),
                         nn.Conv2d(in_channels, out_channels, 1))


class DANet(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(DANet, self).__init__()
        assert in_channels >= 32
        inter_channels = in_channels // 4
        # pam
        self.pam_conv1 = conv2d(in_channels, inter_channels, norm_layer)
        self.pam = PAM_Module(inter_channels)
        self.pam_conv2 = conv2d(inter_channels, inter_channels, norm_layer)
        self.pam_conv3 = dropout_conv1x1(inter_channels, out_channels)

        # cam
        self.cam_conv1 = conv2d(in_channels, inter_channels, norm_layer)
        self.cam = CAM_Module(inter_channels)
        self.cam_conv2 = conv2d(inter_channels, inter_channels, norm_layer)
        self.cam_conv3 = dropout_conv1x1(inter_channels, out_channels)

        self.fusion_conv = dropout_conv1x1(inter_channels, out_channels)

    def forward(self, x):
        p = self.pam_conv1(x)
        p = self.pam(p)
        p = self.pam_conv2(p)

        c = self.cam_conv1(x)
        c = self.cam(c)
        c = self.cam_conv2(c)

        pa = p + c

        pa = self.fusion_conv(pa)
        return pa
