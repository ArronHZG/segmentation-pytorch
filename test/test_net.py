# import torch
#
# from foundation.net.fcn.fcn_4s import fcn
#
# if __name__ == '__main__':
#     x = torch.rand((1, 4, 256, 256))
#
#     m = fcn("fcnResNet50", num_classes=21, in_channels=4, pretrained=False)
#     x = m(x)
#     print(x.shape)
#
# # show_model(m,"fcnResNet50")
import torch

from foundation import UNet
from foundation import count_param

if __name__ == '__main__':
    x = torch.rand((1, 3, 128, 128))
    model = UNet(2, 3)
    param = count_param(model)
    y = model(x)
    print('Output shape:', y.shape)
    print('UNet_Res total parameters: %.2fM (%d)' % (param / 1e6, param))
