# -*- coding: utf-8 -*-
# @Author  : LG
from torch import nn
import numpy as np
import torch
from Utils import corner_form_to_center_form, center_form_to_corner_form

class priorbox:
    """
    Retainnet anchors, 生成策略与SSD不同
    """
    def __init__(self,cfg=None):
        self.features_maps = [(75, 75), (38, 38), (19, 19), (10, 10), (5, 5)]
        self.anchor_sizes = [32, 64, 128, 256, 512]
        self.ratios = np.array([0.5, 1, 2])
        self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
        self.image_size = 600
        self.clip = True
        if cfg:
            self.features_maps = cfg.MODEL.ANCHORS.FEATURE_MAPS
            self.anchor_sizes = cfg.MODEL.ANCHORS.SIZES
            self.ratios = np.array(cfg.MODEL.ANCHORS.RATIOS)
            self.scales = np.array(cfg.MODEL.ANCHORS.SCALES)
            self.image_size = cfg.MODEL.INPUT.IMAGE_SIZE
            self.clip = cfg.MODEL.ANCHORS.CLIP

    def __call__(self):
        priors = []
        for k , (feature_map_w, feature_map_h) in enumerate(self.features_maps):
            for i in range(feature_map_w):
                for j in range(feature_map_h):
                    cx = (j + 0.5) / feature_map_w
                    cy = (i + 0.5) / feature_map_h

                    size = self.anchor_sizes[k]/self.image_size    # 将框体长宽转为 比例形式

                    sides_square = self.scales * size   # 计算方形检测框边长
                    for side_square in sides_square:
                        priors.append([cx, cy, side_square, side_square])   # 添加方形检测框

                    sides_long = sides_square*2**(1/2)  # 计算长形检测框长边
                    for side_long in sides_long:
                        priors.append([cx, cy, side_long, side_long/2]) # 添加长形检测框,短边为长边的一半
                        priors.append([cx, cy, side_long/2, side_long])

        priors = torch.tensor(priors)
        if self.clip:   # 对超出图像范围的框体进行截断
            priors = center_form_to_corner_form(priors) # 截断时,先转为 [xmin, ymin, xmin, xmax]形式
            priors.clamp_(max=1, min=0)
            priors = corner_form_to_center_form(priors) # 转回 [x, y, w, h]形式
        return priors

if __name__ == '__main__':
    anchors = priorbox()()
    print(anchors[-10:])
    print(len(anchors))
