import torch

from foundation.net.FCN.fcn_4s import FCN

if __name__ == '__main__':
    x = torch.rand((1, 4, 256, 256))

    m = FCN("fcnResNet50", num_classes=21, in_channels=4, pretrained=False)
    x = m(x)
    print(x.shape)

# show_model(m,"fcnResNet50")
