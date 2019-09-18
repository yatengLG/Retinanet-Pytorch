# -*- coding: utf-8 -*-
# @Author  : LG

from torch import nn
from torch.nn import functional as F

class fpn(nn.Module):
    def __init__(self,channels_of_fetures, channel_out=256):
        """
        fpn,特征金字塔
        :param channels_of_fetures: list,输入层的通道数,必须与输入特征图相对应
        :param channel_out:
        """
        super(fpn,self).__init__()
        self.channels_of_fetures = channels_of_fetures

        self.lateral_conv1 = nn.Conv2d(channels_of_fetures[2], channel_out, kernel_size=1, stride=1, padding=0)
        self.lateral_conv2 = nn.Conv2d(channels_of_fetures[1], channel_out, kernel_size=1, stride=1, padding=0)
        self.lateral_conv3 = nn.Conv2d(channels_of_fetures[0], channel_out, kernel_size=1, stride=1, padding=0)

        self.top_down_conv1 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1)
        self.top_down_conv2 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1)
        self.top_down_conv3 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        """

        :param features:
        :return:
        """
        c3, c4, c5 = features

        p5 = self.lateral_conv1(c5) # 19
        p4 = self.lateral_conv2(c4) # 38
        p3 = self.lateral_conv3(c3) # 75

        p4 = F.interpolate(input=p5, size=(p4.size(2),p4.size(3)), mode="nearest") + p4
        p3 = F.interpolate(input=p4, size=(p3.size(2),p3.size(3)), mode="nearest") + p3

        p5 = self.top_down_conv1(p5)
        p4 = self.top_down_conv1(p4)
        p3 = self.top_down_conv1(p3)

        return p3, p4, p5