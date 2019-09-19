# -*- coding: utf-8 -*-
# @Author  : LG
import torch.nn as nn
from .Focal_Loss import focal_loss

__all__ = ['multiboxloss']

class multiboxloss(nn.Module):
    def __init__(self, cfg=None, alpha=None, gamma=None, num_classes=None):
        """
        retainnet损失函数,分为类别损失(focal loss)
        框体回归损失(smooth_l1_loss)
        采用分别返回的方式返回.便于训练过程中分析处理
        """
        super(multiboxloss, self).__init__()
        if cfg:
            self.alpha = cfg.MULTIBOXLOSS.ALPHA
            self.gamma = cfg.MULTIBOXLOSS.GAMMA
            self.num_classes = cfg.DATA.DATASET.NUM_CLASSES
        if alpha:
            self.alpha = alpha
        if gamma:
            self.gamma = gamma
        if num_classes:
            self.num_classes = num_classes

        self.loc_loss_fn = nn.SmoothL1Loss(reduction='sum')
        self.cls_loss_fn = focal_loss(alpha=self.alpha, gamma=self.gamma, num_classes=self.num_classes, size_average=False)  # 类别损失为focal loss
        print(" --- Multiboxloss : α={} γ={} num_classes={}".format(self.alpha, self.gamma, self.num_classes))

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

        classification_loss = self.cls_loss_fn(confidence, labels)

        # 回归损失,smooth_l1
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
        gt_locations = gt_locations[pos_mask, :].view(-1, 4)
        smooth_l1_loss = self.loc_loss_fn(predicted_locations, gt_locations)
        num_pos = gt_locations.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos
