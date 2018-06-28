# RetinaNet-PyTorch

## 参考
* [kuangliu/pytorch-retinanet](https://github.com/kuangliu/pytorch-retinanet)  
* [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

## 依赖
* PyTorch-0.3

### Focal Loss
![focal_loss](./imgs/focal_loss.png)
* Focal Loss = weight * Cross-Entropy loss
* delta 是指预测值与真实值之间的差异
* gamma = 0 时, Focal Loss 就是 Cross-Entropy loss


