import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import initialize_weights


class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()

        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # Table 2, best performance with two 3x3 convs
        self.output = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1, stride=1),
        )
        initialize_weights(self)

    def forward(self, x, low_level_features, H, W):
        # low_level_features 变为 channel = 48
        low_level_features = self.low_level_conv(low_level_features)
        # 上采用2/4倍
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)),
                          mode='bilinear',
                          align_corners=True)
        # concat
        x = torch.cat((x, low_level_features), dim=1)
        # 得到结果卷积
        x = self.output(x)
        # 上采用4倍 恢复到原图大小
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        return x
