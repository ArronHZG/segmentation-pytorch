import torch
from torchsummary import summary
from foundation.net.unet.model import ResNet
from foundation.net.unet.resunet import ResUNet


def test_resnet():
    resnet = ResNet(backbone='resnet50',
                    in_channels=3,
                    pretrained=False)

    resnet.cuda()

    x = torch.rand((2, 3, 512, 512)).cuda()

    output0, output1, output2, output3, output4 = resnet(x)

    print(output0.shape)
    print(output1.shape)
    print(output2.shape)
    print(output3.shape)
    print(output4.shape)

    # torch.Size([2, 128, 512, 512])
    # torch.Size([2, 256, 256, 256])
    # torch.Size([2, 512, 128, 128])
    # torch.Size([2, 1024, 64, 64])
    # torch.Size([2, 2048, 32, 32])
def test_resunet():
    # resunet = ResUNet(num_classes = 22, in_channels = 3, backbone = "resnet50", pretrained=False)
    # resunet.cuda()
    # x = torch.rand((2, 3, 512, 512)).cuda()
    # output = resunet(x)
    # print(output.shape)
    resunet = ResUNet(num_classes = 22, in_channels = 3, backbone = "resnet50", pretrained=False)
    resunet.cuda()
    summary(resunet,(3, 256, 256))

test_resunet()