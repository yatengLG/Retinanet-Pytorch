# -*- coding: utf-8 -*-
# @Author  : LG
import torch.nn as nn
import torch.nn.functional as F
import math
from .Focal_Loss import focal_loss
import torch
__all__ = ['multiboxloss']

class multiboxloss(nn.Module):
    def __init__(self, neg_pos_ratio):
        """
        SSD损失函数,分为类别损失(使用cross_entropy)
        框体回归损失(使用smooth_l1_loss)
        这里并没有在返回时,采用分别返回的方式返回.便于训练过程中分析处理
        :param neg_pos_ratio:
        """
        super(multiboxloss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.loc_loss_fn = nn.SmoothL1Loss()
        self.cls_loss_fn = focal_loss(alpha=0.25, gamma=2, num_classes=10)
    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """
            计算类别损失和框体回归损失
        Args:
            confidence (batch_size, num_priors, num_classes): 预测类别
            predicted_locations (batch_size, num_priors, 4): 预测位置
            labels (batch_size, num_priors): 所有框的真实类别
            gt_locations (batch_size, num_priors, 4): 所有框真实的位置
        """
        num_classes = confidence.size(2)

        classification_loss = self.cls_loss_fn(confidence, labels, reduction='sum')

        # 回归损失,smooth_l1
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
        gt_locations = gt_locations[pos_mask, :].view(-1, 4)
        smooth_l1_loss = self.loc_loss_fn(predicted_locations, gt_locations, reduction='sum')
        num_pos = gt_locations.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos
