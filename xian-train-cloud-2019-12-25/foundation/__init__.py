from torch.optim import SGD, Adam

from .blseg.model.pspnet import PSPNet
from .net import *
from .optimizer import RAdam, Lookahead


def get_model(model_name, backbone, num_classes, in_c):
    if model_name == 'deeplabv3plus':
        return DeepLabV3Plus(num_classes, in_channels=in_c, backbone=backbone, pretrained=True)
    if model_name == 'pspnet':
        return PSPNet(backbone=backbone, num_classes=num_classes, in_c=in_c)
    if model_name == 'fcn':
        return FCN(num_classes, in_channels=in_c, backbone=backbone, pretrained=True)
    if model_name == 'fcn-idea':
        return FCN_Idea(num_classes, in_channels=in_c, backbone=backbone, pretrained=True)
    if model_name == 'unet':
        return UNet(num_classes, in_c)
    if model_name == 'unet-nested':
        return UNet_Nested(num_classes, in_channels=in_c)
    if model_name == 'resunet':
        return ResUNet(num_classes, in_channels=in_c, backbone=backbone, pretrained=True)

    raise NotImplementedError


def get_optimizer(optim_name, parameters, lr, lookahead):
    optimizer = None
    if optim_name == 'SGD':
        optimizer =  SGD(parameters, lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    if optim_name == 'Adam':
        optimizer =  Adam(parameters, lr, weight_decay=5e-4)
    if optim_name == 'RAdam':
        optimizer =  RAdam(parameters, lr, weight_decay=5e-4)
    if not optimizer:
        raise NotImplementedError
    if lookahead:
        optimizer = Lookahead(optimizer=optimizer, k=5, alpha=0.5)
    return optimizer

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count
