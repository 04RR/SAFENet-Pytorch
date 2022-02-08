import torch
import torch.nn as nn
import torchvision.transforms as ttf

from utils import CrossPropogationUnit, AffinityPropogationUnit, SELayer


class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, 1, 1)
        self.bn1_s = nn.BatchNorm2d(64)
        self.bn1_d = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.conv2_s = nn.Conv2d(64, 128, 3, 1)
        self.conv2_d = nn.Conv2d(64, 128, 3, 1)
        self.bn2_s = nn.BatchNorm2d(128)
        self.bn2_d = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 1, 1, 1)
        self.bn3_s = nn.BatchNorm2d(256)
        self.bn3_d = nn.BatchNorm2d(256)
        self.se_s = SELayer(256)
        self.se_d = SELayer(256)
        self.conv = nn.Conv2d(3, 256, 3, 1, 1)

    def forward(self, img):

        layer1 = self.conv1(img)

        layer2 = nn.ReLU()(self.bn1_s(layer1)) + nn.ReLU()(self.bn1_d(layer1))

        layer3_o = self.conv2(layer2)
        layer3_s = self.conv2_s(layer2)
        layer3_d = self.conv2_d(layer2)
        layer3 = layer3_o + layer3_s + layer3_d

        layer4 = nn.ReLU()(self.bn2_s(layer3)) + nn.ReLU()(self.bn2_d(layer3))

        layer5 = self.conv3(layer4)

        layer6 = nn.ReLU()(self.bn3_s(layer5)) + nn.ReLU()(self.bn3_d(layer5))

        layer7_s = self.se_s(layer6)
        layer7_d = self.se_d(layer6)

        residue = self.conv(img)
        out = layer7_s + layer7_d + residue

        return out, layer1, layer3_o, layer5


class Decoder(nn.Module):
    def __init__(self, classes, in_channels=256):
        super(Decoder, self).__init__()

        self.conv1_s = nn.Conv2d(in_channels, 256, 3, 1, 1)
        self.conv1_d = nn.Conv2d(in_channels, 256, 3, 1, 1)

        self.cpu1 = CrossPropogationUnit(256)

        self.conv2_s = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv2_d = nn.Conv2d(256, 256, 3, 1, 1)

        self.upconv3_s = nn.ConvTranspose2d(256, 128, 3, 1, 1)
        self.conv3_d = nn.Conv2d(256, 128, 3, 1, 1)

        self.cpu2 = CrossPropogationUnit(128)

        self.conv4_s = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv4_d = nn.Conv2d(128, 128, 3, 1, 1)

        self.apu = AffinityPropogationUnit(128)

        ##

        self.conv6_s = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv6_d = nn.Conv2d(128, 64, 3, 1, 1)

        self.cpu3 = CrossPropogationUnit(64)

        self.conv7_s = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv7_d = nn.Conv2d(64, 64, 3, 1, 1)

        ##

        self.conv9_s = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv9_d = nn.Conv2d(64, 32, 3, 1, 1)

        self.cpu4 = CrossPropogationUnit(32)

        self.conv10_s = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv10_d = nn.Conv2d(32, 32, 3, 1, 1)

        ##

        self.conv12_s = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv12_d = nn.Conv2d(32, 32, 3, 1, 1)

        self.conv13_s = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv13_d = nn.Conv2d(32, 32, 3, 1, 1)

        self.conv14_s = nn.Conv2d(32, classes, 3, 1, 1)
        self.conv14_d = nn.Conv2d(32, 1, 1, 1)

    def forward(self, img_seg, img_depth, skip1, skip2, skip3):
        _, _, x, y = img_depth.shape

        layer1_s, layer1_d = nn.ELU()(self.conv1_s(img_seg)), nn.ELU()(
            self.conv1_d(img_depth)
        )

        layer2_s, layer2_d = self.cpu1(layer1_s, layer1_d)
        # layer2_s, layer2_d = nn.Upsample(4)(layer2_s), nn.Upsample(4)(layer2_d)

        layer3_s, layer3_d = (
            nn.ELU()(self.conv2_s(layer2_s)) + skip3,
            nn.ELU()(self.conv2_d(layer2_d)) + skip3,
        )

        layer4_s, layer4_d = nn.ELU()(self.upconv3_s(layer3_s)), nn.ELU()(
            self.conv3_d(layer3_d)
        )

        layer5_s, layer5_d = self.cpu2(layer4_s, layer4_d)
        # layer5_s, layer5_d = nn.Upsample(4)(layer5_s), nn.Upsample(4)(layer5_d)

        layer6_s, layer6_d = (
            nn.ELU()(self.conv4_s(layer5_s)) + ttf.Resize((x, y))(skip2),
            nn.ELU()(self.conv4_d(layer5_d)) + ttf.Resize((x, y))(skip2),
        )

        layer7 = self.apu(layer6_s, layer6_d)

        layer8_s, layer8_d = nn.ELU()(self.conv6_s(layer7)), nn.ELU()(
            self.conv6_d(layer6_d)
        )

        layer9_s, layer9_d = self.cpu3(layer8_s, layer8_d)
        # layer9_s, layer9_d = nn.Upsample(4)(layer9_s), nn.Upsample(4)(layer9_d)

        layer10_s, layer10_d = (
            nn.ELU()(self.conv7_s(layer9_s)) + skip1,
            nn.ELU()(self.conv7_d(layer9_d)) + skip1,
        )

        layer11_s, layer11_d = nn.ELU()(self.conv9_s(layer10_s)), nn.ELU()(
            self.conv9_d(layer10_d)
        )
        # layer11_s, layer11_d = nn.Upsample(4)(layer11_s), nn.Upsample(4)(layer11_d)

        layer12_s, layer12_d = nn.ELU()(self.conv10_s(layer11_s)), nn.ELU()(
            self.conv10_d(layer11_d)
        )

        layer13_s, layer13_d = nn.ELU()(self.conv12_s(layer12_s)), nn.ELU()(
            self.conv12_d(layer12_d)
        )
        # layer13_s, layer13_d = nn.Upsample(4)(layer13_s), nn.Upsample(4)(layer13_d)

        layer14_s, layer14_d = nn.ELU()(self.conv13_s(layer13_s)), nn.ELU()(
            self.conv13_d(layer13_d)
        )

        out_s, out_d = nn.Sigmoid()(self.conv14_s(layer14_s)), nn.Softmax()(
            self.conv14_d(layer14_d)
        )

        return out_s, out_d


class SAFENet(nn.Module):
    def __init__(self, in_channels, classes):
        super().__init__()

        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(classes= classes)

    def forward(self, img):

        out, skip1, skip2, skip3 = self.encoder(img)

        out = self.decoder(out, out, skip1, skip2, skip3)

        return out
