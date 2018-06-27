import os
import math
import torch

from models.retina import RetinaNet


model_dir = '../preTrainedModels/resnet'
params = torch.load(os.path.join(model_dir, 'resnet50-19c8e357.pth'))

net = RetinaNet(num_classes=21)
net.fpn.load_state_dict(params, strict=False)

torch.nn.init.constant(net.cls_head[-1].bias, -math.log(1-0.01)/0.01)
torch.save(net.state_dict(), os.path.join(model_dir, 'retinanet_resnet50_voc.pth'))
