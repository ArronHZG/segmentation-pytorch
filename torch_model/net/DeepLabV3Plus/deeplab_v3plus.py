from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import getBackBone
from models.aspp import ASSP
from models.decoder import Decoder


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, in_channels=3, backbone='xception', pretrained=True,
                 output_stride=16, freeze_bn=False, **_):
        super(DeepLabV3Plus, self).__init__()
        assert ('xception' or 'resnet' in backbone)
        self.backbone, low_level_channels = getBackBone(backbone, in_channels=in_channels, output_stride=output_stride,
                                                        pretrained=pretrained)

        self.ASSP = ASSP(in_channels=2048, output_stride=output_stride)

        self.decoder = Decoder(low_level_channels, num_classes)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x, low_level_features = self.backbone(x)
        x = self.ASSP(x)
        x = self.decoder(x, low_level_features)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

    # Two functions to yield the parameters of the backbone
    # & Decoder / ASSP to use differentiable learning rates
    # FIXME: in xception, we use the parameters from xception and not aligned xception
    # better to have higher lr for this backbone

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.ASSP.parameters(), self.decoder.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


if __name__ == '__main__':
    m,_ = getBackBone("xception", output_stride=8, pretrained=False)
    print(m)
    x = torch.rand((1, 3, 256, 256))
    print(x.shape)
    a, b = m(x)
    print(a.shape)
    print(b.shape)
