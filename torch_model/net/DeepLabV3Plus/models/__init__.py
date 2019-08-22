import torch
import torch.nn as nn

from .resnet import ResNet
from .xception import Xception


def getBackBone(name, in_channels=3, output_stride=16, pretrained=True):
    model_map = {
        'resnet': DeepLabV3PlusResNet101,
        'xception': DeepLabV3PlusXception
    }

    return model_map[name](in_channels=in_channels, output_stride=output_stride, pretrained=pretrained)


def DeepLabV3PlusResNet101(in_channels=3, output_stride=16, pretrained=True):
    return ResNet(backbone="resnet101", in_channels=in_channels, output_stride=output_stride,
                  pretrained=pretrained), 256


def DeepLabV3PlusXception(in_channels=3, output_stride=16, pretrained=True):
    return Xception(in_channels=in_channels, output_stride=output_stride, pretrained=pretrained), 128


