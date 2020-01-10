import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np

from models.resnet import resnet101

class ASPP(nn.Module):
    def __init__(self, in_channel):
        super(ASPP, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.conv3 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channel)
        self.conv4 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(in_channel)

        self.conv5 = nn.Conv2d(in_channel * 5, in_channel, kernel_size=1, padding=0, dilation=1, bias=False)
        self.bn5 = nn.BatchNorm2d(in_channel)

    def forward(self, x):
        _, _, h, w = x.size()

        x1 = F.interpolate(self.pool(x), size=(h, w), mode='bilinear', align_corners=True)
        x2 = self.bn1(self.conv1(x))
        x3 = self.bn2(self.conv2(x))
        x4 = self.bn3(self.conv3(x))
        x5 = self.bn4(self.conv4(x))

        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.bn5(self.conv5(x))

        return x
   

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=19):
        super(DeepLabV3, self).__init__()
        self.resnet = resnet101(pretrained=True)
        self.aspp = ASPP(2048)
        self.conv = nn.Conv2d(2048, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.resnet(x)
        x = self.aspp(x)
        x = self.conv(x)

        return x
