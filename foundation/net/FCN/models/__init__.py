import torch

from .resnet import ResNet


def getBackBone(name, in_channels, pretrained):
    model_map = {
        'resnet50': fcnResNet50,
        'resnet101': fcnResNet101,
    }

    return model_map[name](in_channels, pretrained)


def fcnResNet50(in_channels, pretrained):
    '''
    Transform resNet50 to fcnResNet50
     specific
     The original resNet output image category,
        the original image is reduced by 32 times after the last big layer,
        and enter avgpool, fc, output image category
     1. Remove the last avgpool, fc layer
     2. Let the models output the output of each layer,
        which is reduced by 4 times, reduced by 8 times,
                 reduced by 16 times, and reduced by 32 times.

    '''
    return ResNet(backbone='resnet50', in_channels=in_channels, pretrained=pretrained)

def fcnResNet101(in_channels, pretrained):
    '''
    Transform resNet50 to fcnResNet50
     specific
     The original resNet output image category,
        the original image is reduced by 32 times after the last big layer,
        and enter avgpool, fc, output image category
     1. Remove the last avgpool, fc layer
     2. Let the models output the output of each layer,
        which is reduced by 4 times, reduced by 8 times,
                 reduced by 16 times, and reduced by 32 times.

    '''
    return ResNet(backbone='resnet101', in_channels=in_channels, pretrained=pretrained)

