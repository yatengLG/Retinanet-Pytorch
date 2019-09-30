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
