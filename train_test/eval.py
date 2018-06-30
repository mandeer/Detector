import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.autograd import Variable
from dataloader.dataaugmentor import DataAugmentor
from dataloader.dataset import DataSet
from evaluations.voc_eval import voc_eval
from models.retinaNet import RetinaNet
from dataloader.en_decoder import RetinaBoxCoder

from PIL import Image


print('Loading model..')
net = RetinaNet(num_classes=21)
checkpoint = torch.load('../output/ckpt.pth')
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()
box_coder = RetinaBoxCoder(imgSize=640)
dataugmentor = DataAugmentor(imgSize=640)

print('Preparing dataset..')
def transform(img, boxes, labels):
    img, boxes = dataugmentor.resize(img, boxes)
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])(img)
    return img, boxes, labels

dataset = DataSet(root='../datasets/voc/VOC2007/JPEGImages',
                  list_file='../datasets/voc//voc07_test.txt',
                  transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

pred_boxes = []
pred_labels = []
pred_scores = []
gt_boxes = []
gt_labels = []

with open('../datasets/voc/voc07_test_difficult.txt') as f:
    gt_difficults = []
    for line in f.readlines():
        line = line.strip().split()
        d = [int(x) for x in line[1:]]
        gt_difficults.append(d)

def eval(net, dataset):
    for i, (inputs, box_targets, label_targets) in enumerate(dataloader):
        print('%d/%d' % (i, len(dataloader)))
        gt_boxes.append(box_targets.squeeze(0))
        gt_labels.append(label_targets.squeeze(0))

        loc_preds, cls_preds = net(Variable(inputs.cuda(), volatile=True))
        box_preds, label_preds, score_preds = box_coder.decode(
            loc_preds.cpu().data.squeeze(),
            cls_preds.cpu().data.squeeze(),
            input_size=[640.0, 640.0])

        pred_boxes.append(box_preds)
        pred_labels.append(label_preds)
        pred_scores.append(score_preds)

    print(voc_eval(
        pred_boxes, pred_labels, pred_scores,
        gt_boxes, gt_labels, gt_difficults,
        iou_thresh=0.5, use_07_metric=True))

eval(net, dataset)
