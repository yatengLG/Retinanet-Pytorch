# -*- coding: utf-8 -*-
# @Author  : LG
from torch import nn
import torch

class predictor(nn.Module):
    """
    Retainnet 分类(cls)及回归(reg)网络
    """
    def __init__(self, num_anchors, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.make_headers()
        self.reset_parameters()

    def forward(self, features):
        """
        对输入的特征图中每个特征点进行分类及回归
        :param features:    # 经过fpn后 输出的特征图
        :return:            # 每个特征点的类别预测与回归预测
        """
        cls_logits = []
        bbox_pred = []
        batch_size = features[0].size(0)
        for feature in features:
            cls_logit = self.cls_headers(feature)
            cls_logits.append(self.cls_headers(feature).permute(0, 2, 3, 1).contiguous().view(batch_size,-1,self.num_classes))
            bbox_pred.append(self.reg_headers(feature).permute(0, 2, 3, 1).contiguous().view(batch_size,-1,4))

        cls_logits = torch.cat(cls_logits, dim=1)
        bbox_pred = torch.cat(bbox_pred, dim=1)

        return cls_logits, bbox_pred

    def make_headers(self):
        cls_headers = []
        reg_headers = []

        for _ in range(4):
            cls_headers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            cls_headers.append(nn.ReLU(inplace=True))

            reg_headers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            reg_headers.append(nn.ReLU(inplace=True))

        cls_headers.append(nn.Conv2d(256, self.num_anchors * self.num_classes, kernel_size=3, stride=1, padding=1))
        reg_headers.append(nn.Conv2d(256, self.num_anchors * 4, kernel_size=3, stride=1, padding=1))

        self.cls_headers = nn.Sequential(*cls_headers)
        self.reg_headers = nn.Sequential(*reg_headers)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
