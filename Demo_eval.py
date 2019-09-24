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