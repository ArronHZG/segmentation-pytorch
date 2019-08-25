import torch
from torch.nn import Module, Conv2d, Parameter, Softmax

__all__ = ['PAM_Module', 'CAM_Module']


class PAM_Module(Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_channels):
        super(PAM_Module, self).__init__()
        inter_channels = in_channels // 8

        self.query_conv = Conv2d(in_channels, inter_channels, kernel_size=1)
        self.key_conv = Conv2d(in_channels, inter_channels, kernel_size=1)
        self.value_conv = Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B,C,H,W)
            returns :
                out : attention value + input feature
                attention: B,(HxW),(HxW)
        """
        B, C, H, W = x.size()
        # query (B,WH,C//8)
        query = self.query_conv(x).reshape(B, -1, H * W).permute(0, 2, 1)
        # key (B,C//8,HW)
        key = self.key_conv(x).reshape(B, -1, H * W)
        # value (B,C,HW)
        value = self.value_conv(x).reshape(B, -1, H * W)

        # energy (B,WH,WH)
        energy = torch.bmm(query, key)
        # attention (B,WH,WH)
        attention = self.softmax(energy)
        # print(attention[0][0]) #[0.1114, 0.1236, 0.1127, 0.1003, 0.1041, 0.1065, 0.1078, 0.1163, 0.1173]
        # print(sum(attention[0][0])) # 1
        # out (B,C,WH)
        out = torch.bmm(value, attention)
        # out = torch.bmm(value, attention.permute(0, 2, 1))
        # out (B,C,W,H)
        out = out.reshape(B, C, H, W)

        out = self.gamma * out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B,C,H,W)
            returns :
                out : attention value + input feature
                attention: (B,C,C)
        """
        B, C, H, W = x.size()
        # query (B,C,HW)
        query = x.reshape(B, C, -1)
        # key (B,HW,C)
        key = x.reshape(B, C, -1).permute(0, 2, 1)
        # value (B,C,HW)
        value = x.reshape(B, C, -1)

        # energy (B,C,C)
        energy = torch.bmm(query, key)
        # max_energy (B,C,C)
        max_energy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
        energy = max_energy - energy
        attention = self.softmax(energy)
        out = torch.bmm(attention, value)
        out = out.reshape(B, C, H, W)

        out = self.gamma * out + x
        return out


if __name__ == '__main__':
    a = torch.rand((2, 2, 3, 3))
    m = CAM_Module(2)
    o = m(a)
    print(o.shape)
