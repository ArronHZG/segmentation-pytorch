import torch
import torch.nn as nn

from . import torchvision_resnet
from .utils import initialize_weights


class ResNet(nn.Module):
    def __init__(self,
                 backbone='resnet50',
                 in_channels=3,
                 pretrained=True):
        super(ResNet, self).__init__()
        model = getattr(torchvision_resnet, backbone)(pretrained)

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )
        self.layer0_5 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        initialize_weights(self.layer0)

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if not pretrained:
            initialize_weights(self)
            for m in self.modules():
                if isinstance(m, torchvision_resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, torchvision_resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.layer0(x)
        output0 = x
        x=self.layer0_5(x)
        x = self.layer1(x)
        output1 = x
        x = self.layer2(x)
        output2 = x
        x = self.layer3(x)
        output3 = x
        output4 = self.layer4(x)

        return output0, output1, output2, output3, output4
