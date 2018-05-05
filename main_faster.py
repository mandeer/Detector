# -*- coding: utf-8 -*-
import argparse
import os
import random
import time
from collections import namedtuple
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as Func
from dataLoader import getVocDataLoader
from models.Faster import FasterRCNNVGG16, fast_rcnn_loc_loss
from models.Faster.utils import AnchorTargetCreator, ProposalTargetCreator
from utils.vis_tool import Visualizer
import utils.array_tool as at
from torchnet.meter import ConfusionMeter, AverageValueMeter


LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'])


def get_optimizer(config, faster_rcnn):
    """
    return optimizer, It could be overwriten if you want to specify
    special optimizer
    """
    lr = config.lr
    params = []
    for key, value in dict(faster_rcnn.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': config.weight_decay}]
    if config.use_adam:
        optimizer = torch.optim.Adam(params)
    else:
        optimizer = torch.optim.SGD(params, momentum=0.9)
    return optimizer


class FasterRCNNTrainer(nn.Module):
    """wrapper for conveniently training. return losses

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.
    * :obj:`total_loss`: The sum of 4 loss above.

    Args:
        faster_rcnn (model.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
    """

    def __init__(self, config, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()
        self.config = config

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = 3  # opt.rpn_sigma
        self.roi_sigma = 1  # opt.roi_sigma

        # target creator create gt_bbox gt_label etc as training targets.
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = get_optimizer(config=config, faster_rcnn=faster_rcnn)
        # visdom wrapper
        self.vis = Visualizer(env='faster-rcnn')

        # indicators for training status
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    def forward(self, imgs, bboxes, labels, scale):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~torch.autograd.Variable): A variable with a batch of images.
            bboxes (~torch.autograd.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~torch.autograd..Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground classes.
            scale (float): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            namedtuple of 5 losses
        """
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois,
        # consider them as constant input
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = torch.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)

        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            img_size)
        gt_rpn_label = at.tovariable(gt_rpn_label).long()
        gt_rpn_loc = at.tovariable(gt_rpn_loc)
        rpn_loc_loss = fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label.data, self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        rpn_cls_loss = Func.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(), at.totensor(gt_roi_label).long()]
        gt_roi_label = at.tovariable(gt_roi_label).long()
        gt_roi_loc = at.tovariable(gt_roi_loc)

        roi_loc_loss = fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

        self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.

        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = self.config._state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        torch.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = torch.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            self.config._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}


def main(config):
    # cuda
    if config.use_cuda:
        from torch.backends import cudnn
        cudnn.benchmark = True
    elif torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # seed
    if config.seed == 0:
        config.seed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.use_cuda:
        torch.cuda.manual_seed_all(config.seed)

    # create directories if not exist
    if not os.path.exists(config.out_path):
        os.makedirs(config.out_path)

    # data
    if config.dataset == 'COCO':
        config.data_path = './dataset/coco'
        config.num_classes = 80
    elif config.dataset == 'VOC07':
        config.data_path = './dataset/voc07/VOCdevkit/VOC2007'
        config.num_classes = 20
        trainLoader, valLoader = getVocDataLoader(config)
        print('train samples num: ', len(trainLoader), '  test samples num: ', len(valLoader))
    else:
        print('Only support Coco and VOC!!')
        return

    faster_rcnn = FasterRCNNVGG16(config)
    print(faster_rcnn)
    # trainer = FasterRCNNTrainer(faster_rcnn).cuda()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image-size',     type=int,      default=32)
    parser.add_argument('--n-epochs',       type=int,      default=50)
    parser.add_argument('--batch-size',     type=int,      default=1)
    parser.add_argument('--n-workers',      type=int,      default=4)
    parser.add_argument('--lr',             type=float,    default=0.1)
    parser.add_argument('--weight-decay',   type=float,    default=0.0005)
    parser.add_argument('--out-path',       type=str,      default='./output')
    parser.add_argument('--seed',           type=int,      default=0,           help='random seed for all')
    parser.add_argument('--log-step',       type=int,      default=100)
    parser.add_argument('--use-cuda',       type=bool,     default=True,        help='enables cuda')

    parser.add_argument('--dataset',        type=str,      default='VOC07',      help='COCO or VOC07, VOC12')
    parser.add_argument('--mode',           type=str,      default='train',     help='train, test')
    parser.add_argument('--model',          type=str,      default='VGG',       help='model')
    parser.add_argument('--pretrained',     type=str,      default='./preTrainedModels/vgg/vgg16.pth')

    config = parser.parse_args()
    if config.use_cuda and not torch.cuda.is_available():
        config.use_cuda = False
        print("WARNING: You have no CUDA device")

    args = vars(config)
    print('------------ Options -------------')
    for key, value in sorted(args.items()):
        print('%16.16s: %16.16s' % (str(key), str(value)))
    print('-------------- End ----------------')

    main(config)
    print('End!!')