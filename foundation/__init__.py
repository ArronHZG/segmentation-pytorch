from torch.optim import SGD, Adam

from .blseg.model.pspnet import PSPNet
from .net import *
from .optimizer.radam import RAdam


def get_model(model_name, backbone, num_classes, in_c):
    if model_name == 'deeplabv3plus':
        return DeepLabV3Plus(num_classes, in_channels=in_c, backbone=backbone, pretrained=True)
    if model_name == 'PSPNet':
        return PSPNet(backbone=backbone, num_classes=num_classes, in_c=in_c)
    if model_name == 'fcn':
        return FCN_Idea(num_classes, in_channels=in_c, backbone=backbone, pretrained=True)
    if model_name == 'fcn-idea':
        return FCN_(num_classes, in_channels=in_c, backbone=backbone, pretrained=True)
    if model_name == 'unet':
        return UNet(num_classes, in_c)
    if model_name == 'UNet_Nested':
        return UNet_Nested(num_classes, in_channels=in_c)
    if model_name == 'resunet':
        return ResUNet(num_classes, in_channels=in_c, backbone=backbone, pretrained=True)

    raise NotImplementedError


def get_optimizer(optim_name, parameters, lr):
    if optim_name == 'SGD':
        return SGD(parameters, lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    if optim_name == 'Adam':
        return Adam(parameters, lr, weight_decay=5e-4)
    if optim_name == 'RAdam':
        return RAdam(parameters, lr, weight_decay=5e-4)
    raise NotImplementedError


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count
