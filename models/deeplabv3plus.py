import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np

from models.resnet import resnet101
from inplace_abn import InPlaceABNSync

class ASPP(nn.Module):
    def __init__(self, in_channel):
        super(ASPP, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.conv1 = nn.Conv2d(in_channel, 256, kernel_size=1, padding=0, dilation=1, bias=False)
        self.bn1 = InPlaceABNSync(256)
        self.conv2 = nn.Conv2d(in_channel, 256, kernel_size=1, padding=0, dilation=1, bias=False)
        self.bn2 = InPlaceABNSync(256)
        self.conv3 = nn.Conv2d(in_channel, 256, kernel_size=3, padding=6, dilation=6, bias=False)
        self.bn3 = InPlaceABNSync(256)
        self.conv4 = nn.Conv2d(in_channel, 256, kernel_size=3, padding=12, dilation=12, bias=False)
        self.bn4 = InPlaceABNSync(256)
        self.conv5 = nn.Conv2d(in_channel, 256, kernel_size=3, padding=18, dilation=18, bias=False)
        self.bn5 = InPlaceABNSync(256)

        self.conv6 = nn.Conv2d(256 * 5, 256, kernel_size=1, padding=0, dilation=1, bias=False)
        self.bn6 = InPlaceABNSync(256)

        self.relu = nn.ReLU(inplace=False)
        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        batch, _, h, w = x.size()

        if batch > 1:
            x1 = self.relu(self.bn1(self.conv1(self.pool(x))))
        else:
            x1 = self.relu(self.conv1(self.pool(x)))
        x1 = F.interpolate(x1, size=(h, w), mode='bilinear')
        x2 = self.relu(self.bn2(self.conv2(x)))
        x3 = self.relu(self.bn3(self.conv3(x)))
        x4 = self.relu(self.bn4(self.conv5(x)))
        x5 = self.relu(self.bn5(self.conv5(x)))

        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.drop(self.relu(self.bn6(self.conv6(x))))

        return x
   

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=19, os=16):
        super(DeepLabV3Plus, self).__init__()
        self.resnet = resnet101(os=os, pretrained=True)
        self.aspp = ASPP(2048)
        self.conv1 = nn.Conv2d(256, 48, kernel_size=1, padding=0)
        self.bn1 = InPlaceABNSync(48)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(304, 256, kernel_size=3, padding=1)
        self.bn2 = InPlaceABNSync(256)
        self.conv3 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0)


    def forward(self, x):
        x, low_level_feature = self.resnet(x)
        x = self.aspp(x)

        low_level_feature = self.relu(self.bn1(self.conv1(low_level_feature)))

        x = F.interpolate(x, size=low_level_feature.size()[2:], mode='bilinear')
        x = torch.cat((x, low_level_feature), dim=1)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)

        return x
