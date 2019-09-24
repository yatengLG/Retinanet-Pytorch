# Retinanet-Pytorch

[Retinanet](https://arxiv.org/abs/1708.02002)目标检测算法pytorch实现,

由于一些原因,训练已经过测试,但是并没有训练完毕,所以不会上传预训练模型.

但项目代码验证无误.
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
| | ssd_model|SSD模型类 |
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
