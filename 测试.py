from Model import RetainNet

net = RetainNet('resnet50').cuda()

import torch
a = torch.randn([1,3,600,600]).cuda()
cls_logits, bbox_pred = net(a)

print(cls_logits.size())
print(bbox_pred.size())

det = net.forward_with_postprocess(a)
print(det)