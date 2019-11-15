# -*- coding: utf-8 -*-
# @Author  : LG

from .base_models import build_resnet
from .struct import fpn, predictor, postprocessor
from torch import nn
from Data.Transfroms import transfrom
from vizer.draw import draw_boxes
import torch
from PIL import Image
import numpy as np
import time

class RetainNet(nn.Module):
    """
    :return cls_logits, torch.Size([C, 67995, num_classes])
            bbox_pred,  torch.Size([C, 67995, 4])
    """
    def __init__(self,cfg=None, resnet=None):
        super(RetainNet,self).__init__()
        self.resnet = 'resnet50'
        self.num_classes = 21
        self.num_anchors = 9
        self.cfg = cfg
        if cfg:
            self.resnet = cfg.MODEL.BASEMODEL
            self.num_classes = cfg.DATA.DATASET.NUM_CLASSES
            self.num_anchors = cfg.MODEL.ANCHORS.NUMS
        if resnet:
            self.resnet = resnet
        expansion_list={
            'resnet18': 1,
            'resnet34': 1,
            'resnet50': 4,
            'resnet101': 4,
            'resnet152': 4,
        }
        assert self.resnet in expansion_list

        self.backbone = build_resnet(self.resnet, pretrained=True)
        expansion = expansion_list[self.resnet]
        self.fpn = fpn(channels_of_fetures=[128*expansion, 256*expansion, 512*expansion])
        self.predictor = predictor(num_anchors=self.num_anchors, num_classes=self.num_classes)  # num_anchors 默认为9,与anchor生成相对应
        self.postprocessor = postprocessor(cfg)
        
    def load_pretrained_weight(self, weight_pkl):
        self.load_state_dict(torch.load(weight_pkl))
        
    def forward(self, x):
        c3, c4, c5, p6, p7 = self.backbone(x)   # resnet输出五层特征图
        p3, p4, p5 = self.fpn([c3, c4, c5])     # 前三层特征图进FPN
        features = [p3, p4, p5, p6, p7]
        cls_logits, bbox_pred = self.predictor(features)
        return cls_logits, bbox_pred

    def forward_with_postprocess(self, images):
        """
        前向传播并后处理
        :param images:
        :return:
        """
        cls_logits, bbox_pred = self.forward(images)
        detections = self.postprocessor(cls_logits, bbox_pred)
        return detections

    @torch.no_grad()
    def Detect_single_img(self, image, score_threshold=0.7, device='cuda'):
        """
        检测单张照片
        eg:
            image, boxes, labels, scores= net.Detect_single_img(img)
            plt.imshow(image)
            plt.show()

        :param image:           图片,PIL.Image.Image
        :param score_threshold: 阈值
        :param device:          检测时所用设备,默认'cuda'
        :return:                添加回归框的图片(np.array),回归框,标签,分数
        """
        self.eval()
        assert isinstance(image, Image.Image)
        w, h = image.width, image.height
        images_tensor = transfrom(self.cfg, is_train=False)(np.array(image))[0].unsqueeze(0)

        self.to(device)
        images_tensor = images_tensor.to(device)
        time1 = time.time()
        detections = self.forward_with_postprocess(images_tensor)[0]
        boxes, labels, scores = detections
        boxes, labels, scores = boxes.to('cpu').numpy(), labels.to('cpu').numpy(), scores.to('cpu').numpy()
        boxes[:, 0::2] *= (w / self.cfg.MODEL.INPUT.IMAGE_SIZE)
        boxes[:, 1::2] *= (h / self.cfg.MODEL.INPUT.IMAGE_SIZE)

        indices = scores > score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        print("Detect {} object, inference cost {:.2f} ms".format(len(scores), (time.time() - time1) * 1000))
        # 图像数据加框
        drawn_image = draw_boxes(image=image, boxes=boxes, labels=labels,
                                 scores=scores, class_name_map=self.cfg.DATA.DATASET.CLASS_NAME).astype(np.uint8)
        return drawn_image, boxes, labels, scores

    @torch.no_grad()
    def Detect_video(self, video_path, score_threshold=0.5, save_video_path=None, show=True):
        """
        检测视频
        :param video_path:      视频路径  eg: /XXX/aaa.mp4
        :param score_threshold:
        :param save_video_path: 保存路径,不指定则不保存
        :param show:            在检测过程中实时显示,(会存在卡顿现象,受检测效率影响)
        :return:
        """
        import cv2
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        weight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if save_video_path:
            out = cv2.VideoWriter(save_video_path, fourcc, cap.get(5), (weight, height))
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                drawn_image, boxes, labels, scores = self.Detect_single_img(image=image,
                                                                            device='cuda:0',
                                                                            score_threshold=score_threshold)
                frame = cv2.cvtColor(np.asarray(drawn_image), cv2.COLOR_RGB2BGR)
                if show:
                    cv2.imshow('frame', frame)
                if save_video_path:
                    out.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        if save_video_path:
            out.release()
        cv2.destroyAllWindows()
        return True
