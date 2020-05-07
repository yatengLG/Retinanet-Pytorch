GIthub使用指北:

**1.想将项目拷贝到自己帐号下就fork一下.**

**2.持续关注项目更新就star一下**

**3.watch是设置接收邮件提醒的.**

---

# Retinanet-Pytorch

[Retinanet](https://arxiv.org/abs/1708.02002)目标检测算法pytorch实现,

**本项目不是完全的复现论文**（很多参数以及实现方式上与原论文存在部分差异，有疑问欢迎issues）

由于一些原因,训练已经过测试,但是并没有训练完毕,所以不会上传预训练模型.

但项目代码验证无误.（但在使用时需要自己进行调整。不建议新手进行尝试。）
***
项目在架构上与 [SSD-Pytorch](https://github.com/yatengLG/SSD-Pytorch) 采用了相似的结构.

重用了大量[SSD-Pytorch](https://github.com/yatengLG/SSD-Pytorch)中代码,如训练器,测试器等.

***

**本项目单机多卡,通过torch.nn.DataParallel实现,将单机环境统一包装.支持单机单卡,单机多卡,指定gpu训练及测试,但不支持多机多卡和cpu训练和测试.
不限定检测时的设备(cpu,gpu均可).**

***

# Requirements


1. pytorch
2. opencv-python
3. torchvision >= 0.3.0
4. Vizer
5. visdom

(均可pip安装)

## 项目结构

| 文件夹| 文件 |说明 |
|:-------:|:-------:|:------:|
| **Data** | 数据相关 |
| | Dataloader| 数据加载器类'Our_Dataloader', 'Our_Dataloader_test'|
| | Dataset_VOC|VOC格式数据集类|
| | Transfroms|数据Transfroms|
| | Transfroms_tuils|Transfroms子方法|
| **Model**| 模型相关|
| | base_models/Resnet|支持resnet18,34,50,101,152|
| | structs/Anchors|retinanet默认检测框生成器|
| | structs/MutiBoxLoss|损失函数|
| | structs/Focal_Loss|focal_loss损失函数|
| | structs/Fpn|特征金字塔结构|
| | structs/PostProcess|后处理|
| | structs/Predictor|分类及回归网络|
| | evaler |验证器,用于在数据集上对模型进行验证(测试),计算ap,map |
| | retainnet|Retinanet模型类 |
| | trainer|训练器,用于在数据集上训练模型 |
| **Utils**|各种工具|
| |boxs_op |各种框体操作,编码解码,IOU计算,框体格式转换等|
| **Weights**| 模型权重存放处|
| | pretrained|预训练模型权重存放处,本项目模型并没有训练完毕,因而没有上传训练好的模型,但是训练过程已经过验证|
| | trained |训练过程中默认模型存放处|
| ---- | Configs.py|配置文件,包含了模型定义,数据以及训练过程,测试过程等的全部参数,建议备份一份再进行修改|
| ---- | Demo_train.py| 模型训练的例子,训练过程中的模型会保存在Weights/Our/ |
| ---- | Demo_eval.py| 模型测试的例子,计算模型ap,map |
| ---- | Demo_detect_one_image.py|检测单张图片例子|
| ---- | Demo_detect_video.py|视频检测例子,传入一个视频,进行检测|


# Demo

本项目配有训练,验证,检测部分的代码,所有Demo均经过测试,可直接运行.

## 训练train

同[针对单机多卡环境的SSD目标检测算法实现(Single Shot MultiBox Detector)(简单,明了,易用,中文注释)](https://ptorch.com/news/252.html)一样,项目**使用visdom进行训练过程可视化**.在运行前请安装并运行visdom.

同样的,训练过程也只支持单机单卡或单机多卡环境,不支持cpu训练.

```python

# -*- coding: utf-8 -*-
# @Author  : LG

from Model import RetainNet, Trainer
from Data import vocdataset
from Configs import _C as cfg
from Data import transfrom,targettransform


# 训练数据集,VOC格式数据集, 训练数据取自 ImageSets/Main/train.txt'
train_dataset=vocdataset(cfg, is_train=True, transform=transfrom(cfg,is_train=True),
                         target_transform=targettransform(cfg))

# 测试数据集,VOC格式数据集, 测试数据取自 ImageSets/Main/eval.txt'
test_dataset = vocdataset(cfg=cfg, is_train=False,
                          transform=transfrom(cfg=cfg, is_train=False),
                          target_transform=targettransform(cfg))

if __name__ == '__main__':
    """
    使用时,请先打开visdom
    
    命令行 输入  pip install visdom          进行安装 
    输入        python -m visdom.server'    启动
    """
  
    # 首次调用会下载resnet预训练模型
    
    # 实例化模型. 模型的具体各种参数在Config文件中进行配置
    net = RetainNet(cfg)
    # 将模型移动到gpu上,cfg.DEVICE.MAINDEVICE定义了模型所使用的主GPU
    net.to(cfg.DEVICE.MAINDEVICE)
    # 初始化训练器,训练器参数通过cfg进行配置;也可传入参数进行配置,但不建议
    trainer = Trainer(cfg)
    # 训练器开始在 数据集上训练模型
    trainer(net, train_dataset)
```

## 验证eval
验证过程支持单机多卡,单机单卡,不支持cpu.

```python
# -*- coding: utf-8 -*-
# @Author  : LG

from Model import RetainNet, Evaler
from Data import vocdataset
from Configs import _C as cfg
from Data import transfrom,targettransform


# 训练数据集,VOC格式数据集, 训练数据取自 ImageSets/Main/train.txt'
train_dataset=vocdataset(cfg, is_train=True, transform=transfrom(cfg,is_train=True),
                         target_transform=targettransform(cfg))

# 测试数据集,VOC格式数据集, 测试数据取自 ImageSets/Main/eval.txt'
test_dataset = vocdataset(cfg=cfg, is_train=False,
                          transform=transfrom(cfg=cfg, is_train=False),
                          target_transform=targettransform(cfg))

if __name__ == '__main__':
    # 模型测试只支持GPU单卡或多卡,不支持cpu
    net = RetainNet(cfg)
    # 将模型移动到gpu上,cfg.DEVICE.MAINDEVICE定义了模型所使用的主GPU
    net.to(cfg.DEVICE.MAINDEVICE)
    # 模型从权重文件中加载权重
    net.load_pretrained_weight('XXX.pkl')
    # 初始化验证器,验证器参数通过cfg进行配置;也可传入参数进行配置,但不建议
    evaler = Evaler(cfg, eval_devices=None)
    # 验证器开始在数据集上验证模型
    ap, map = evaler(model=net,
                     test_dataset=test_dataset)
    print(ap)
    print(map)
```

## 检测Detect

单次检测过程支持单机单卡,cpu.

### 单张图片检测

```python
# -*- coding: utf-8 -*-
# @Author  : LG
from Model import RetainNet
from Configs import _C as cfg
from PIL import Image
from matplotlib import pyplot as plt

# 实例化模型
net = RetainNet(cfg)
# 使用cpu或gpu
net.to('cuda')
# 模型从权重文件中加载权重
net.load_pretrained_weight('XXX.pkl')
# 打开图片
image = Image.open("XXX.jpg")
# 进行检测, 分别返回 绘制了检测框的图片数据/回归框/标签/分数.
drawn_image, boxes, labels, scores = net.Detect_single_img(image=image,score_threshold=0.5)

plt.imsave('XXX_det.jpg',drawn_image)
plt.imshow(drawn_image)
plt.show()
```

### 视频检测

```python
# -*- coding: utf-8 -*-
# @Author  : LG
from Model import RetainNet
from Configs import _C as cfg

# 实例化模型
net = RetainNet(cfg)
# 使用cpu或gpu
net.to('cuda')
# 模型从权重文件中加载权重
net.load_pretrained_weight('XXX.pkl')

video_path = 'XXX.mp4'

# 进行检测,
# if save_video_path不为None,则不保存视频,如需保存视频save_video_path=XXX.mp4 ,
# show=True,实时显示检测结果
net.Detect_video(video_path=video_path, score_threshold=0.02, save_video_path=None, show=True)

```

---

support by **jetbrains**.

<img width='300' alt='Jetbrains' src='https://github.com/yatengLG/SSD-Pytorch/blob/master/Images/jetbrains-variant-3.png'>

https://www.jetbrains.com/?from=SSD-Pytorch

---
