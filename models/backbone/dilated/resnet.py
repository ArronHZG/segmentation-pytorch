"""Dilated ResNet"""
import math
import torch.nn as nn

from models.nn.se_model import SELayer
from utils.mypath import Path

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'BasicBlock', 'Bottleneck']


class BasicBlock(nn.Module):
    """ResNet BasicBlock
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1,
                 norm_layer=None, se=False,
                 non_local=False,
                 gc=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = se
        self.non_local = non_local
        self.gc = gc
        self.se_layer = SELayer(inplanes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.se:
            out = self.se_layer(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None,
                 se=False,
                 non_local=False,
                 gc=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        self.se = se
        self.non_local = non_local
        self.gc = gc
        self.se_layer = SELayer(inplanes*self.expansion)

    def _sum_each(self, x, y):
        assert (len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
        return z

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.se:
            out = self.se_layer(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """Dilated Pre-trained ResNet Model, which preduces the stride of 8 featuremaps at conv5.

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).

    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    # pylint: disable=unused-variable
    def __init__(self, block, layers, num_classes=1000, dilated=32,
                 BatchNorm=nn.BatchNorm2d,
                 multi_grid=False,
                 multi_dilation=None,
                 se=False,
                 non_local=False,
                 gc=False):
        self.se = se
        self.non_local = non_local
        self.gc = gc
        if dilated == 8:
            stride = [1, 2, 1, 1]
            dilation = [1, 1, 2, 4]
        elif dilated == 16:
            stride = [1, 2, 2, 1]
            dilation = [1, 1, 1, 2]
        elif dilated == 32:
            stride = [1, 2, 2, 2]
            dilation = [1, 1, 1, 1]
        else:
            raise NotImplementedError

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=stride[0], dilation=dilation[0],
                                       norm_layer=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=stride[1], dilation=dilation[1],
                                       norm_layer=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride[2], dilation=dilation[2],
                                       norm_layer=BatchNorm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=stride[3], dilation=dilation[3],
                                       norm_layer=BatchNorm, multi_grid=multi_grid,
                                       multi_dilation=multi_dilation)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, multi_grid=False,
                    multi_dilation=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []

        if multi_grid:
            layers.append(block(self.inplanes, planes, stride, dilation=multi_dilation[0],
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer,
                                se=self.se,
                                non_local=self.non_local,
                                gc=self.gc))
            self.inplanes = planes * block.expansion
            div = len(multi_dilation)
            for i in range(1, blocks):
                layers.append(
                    block(self.inplanes, planes, dilation=multi_dilation[i % div], previous_dilation=dilation,
                          norm_layer=norm_layer,
                          se=self.se,
                          non_local=self.non_local,
                          gc=self.gc))
        else:
            if dilation == 1 or dilation == 2:
                layers.append(block(self.inplanes, planes, stride, dilation=1,
                                    downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer,
                                    se=self.se,
                                    non_local=self.non_local,
                                    gc=self.gc))
            elif dilation == 4:
                layers.append(block(self.inplanes, planes, stride, dilation=2,
                                    downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer,
                                    se=self.se,
                                    non_local=self.non_local,
                                    gc=self.gc))
            else:
                raise RuntimeError("=> unknown dilation size: {}".format(dilation))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation,
                                    norm_layer=norm_layer,
                                    se=self.se,
                                    non_local=self.non_local,
                                    gc=self.gc))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool( x )
        # x = x.view( x.size( 0 ), -1 )
        # x = self.fc( x )

        return x


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        hub_model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        model.load_state_dict(hub_model.state_dict(), strict=False)
    return model


def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        hub_model = torch.hub.load('pytorch/vision', 'resnet34', pretrained=True)
        model.load_state_dict(hub_model.state_dict(), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        hub_model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
        model.load_state_dict(hub_model.state_dict(), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        hub_model = torch.hub.load('pytorch/vision', 'resnet101', pretrained=True)
        model.load_state_dict(hub_model.state_dict(), strict=False)
    return model


def resnet152(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        hub_model = torch.hub.load('pytorch/vision', 'resnet152', pretrained=True)
        model.load_state_dict(hub_model.state_dict(), strict=False)
    return model


if __name__ == "__main__":
    import torch
    from pprint import pprint
    from utils.model_size import show_model

    model = resnet50(pretrained=True,se=True)
    input = torch.rand(1, 3, 512, 512)
    output = model(input)
    print(output.size())
    show_model(model, "resnet50", input_zise=(3, 512, 512))

    # entrypoints = torch.hub.list('pytorch/vision', force_reload=True)
    # pprint(entrypoints)

    '''['alexnet',
 'deeplabv3_resnet101',
 'densenet121',
 'densenet161',
 'densenet169',
 'densenet201',
 'fcn_resnet101',
 'googlenet',
 'inception_v3',
 'mobilenet_v2',
 'resnet101',
 'resnet152',
 'resnet18',
 'resnet34',
 'resnet50',
 'resnext101_32x8d',
 'resnext50_32x4d',
 'shufflenet_v2_x0_5',
 'shufflenet_v2_x1_0',
 'squeezenet1_0',
 'squeezenet1_1',
 'vgg11',
 'vgg11_bn',
 'vgg13',
 'vgg13_bn',
 'vgg16',
 'vgg16_bn',
 'vgg19',
 'vgg19_bn',
 'wide_resnet101_2',
 'wide_resnet50_2']'''
