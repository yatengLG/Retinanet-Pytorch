from Data import vocdataset,our_dataloader,transfrom,targettransform
from Configs import _C as cfg
from Model import RetainNet
import torch

# dataset = vocdataset(cfg, is_train=True,transform=transfrom(cfg,is_train=True),target_transform=targettransform(cfg))
# dataloader = our_dataloader(dataset,batch_size=10, shuffle=False)
#
# for i, (image, boxes, labels, image_name) in enumerate(dataloader):
#     print(image.size())
#     print(boxes.size())
#     print(labels.size())
#     if i ==1:
#         break
import torchvision.models.resnet
net = RetainNet(cfg).cuda()
x = torch.randn((2,3,600,600)).cuda()
y = net.forward_with_postprocess(x)
print(y)