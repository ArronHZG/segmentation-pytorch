import torch

from torch_model.net.DeepLabV3Plus.models import ResNet
from torch_model.net.FCN.fcn_4s import FCN

if __name__ == '__main__':
    x = torch.rand((1, 4, 256, 256))

    m = ResNet(output_stride=16, in_channels=4, pretrained=False)
    x, y = m(x)
    print(x.shape)
    print(y.shape)

# x = torch.rand((1, 4, 256, 256))
# output_stride = 8
# torch.Size([1, 2048, 32, 32]) out_stride = 8
# torch.Size([1, 256, 64, 64])  out_stride = 4

# x = torch.rand((1, 4, 256, 256))
# output_stride = 16
# torch.Size([1, 2048, 16, 16]) out_stride = 16
# torch.Size([1, 256, 64, 64])  out_stride = 4

# show_model(m,"fcnResNet50")
