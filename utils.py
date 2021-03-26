import torch
import torch.nn as nn


class CrossPropogationUnit(nn.Module):
    def __init__(self, in_channels, kernal=3, stride=1, padding=0):
        super(CrossPropogationUnit, self).__init__()

        self.conv_s_1 = nn.Conv2d(in_channels, in_channels, kernal, stride, 1)
        self.conv_d_1 = nn.Conv2d(in_channels, in_channels, kernal, stride, 1)
        self.conv_s_2 = nn.Conv2d(in_channels, in_channels, kernal, stride, 1)
        self.conv_d_2 = nn.Conv2d(in_channels, in_channels, kernal, stride, 1)

    def forward(self, img_seg, img_depth):
        out_d_1 = self.conv_d_1(img_depth)
        out_d_2 = self.conv_d_2(img_seg)

        out_s_1 = self.conv_s_1(img_seg)
        out_s_2 = self.conv_s_2(img_depth)

        out_depth = out_d_1 + out_d_2 + img_depth
        out_seg = out_s_1 + out_s_2 + img_seg

        return out_depth, out_seg


class AffinityPropogationUnit(nn.Module):
    def __init__(self, in_channels, kernal=1, stride=1, padding=0):
        super(AffinityPropogationUnit, self).__init__()

        self.conv_G = nn.Conv2d(in_channels, in_channels, kernal, stride, padding)
        self.conv_F = nn.Conv2d(in_channels, in_channels, kernal, stride, padding)
        self.conv_K = nn.Conv2d(in_channels, in_channels, kernal, stride, padding)

        self.batchnorm = nn.BatchNorm2d(in_channels)

    def forward(self, img_seg, img_depth):

        out_K = self.conv_K(img_seg)
        n, c, h, w = out_K.shape
        out_K = torch.reshape(out_K, (n, c, h * w))

        out_F = self.conv_F(img_seg)
        n, c, h, w = out_F.shape
        out_F = torch.reshape(out_F, (n, h * w, c))

        out_S = torch.reshape(torch.matmul(out_F, out_K), (n, h * w, h * w))

        out_G = self.conv_G(img_depth)
        n, c, h, w = out_G.shape
        out_G = torch.reshape(out_G, (n, h * w, c))

        out = torch.einsum("nxc,nxx->nxc", out_G, out_S)
        out = torch.reshape(out, (n, c, h, w))
        out = self.batchnorm(out)
        out = out + img_depth

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)