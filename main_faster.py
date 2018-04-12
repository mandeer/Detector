# -*- coding: utf-8 -*-
import argparse
import os
import random
import torch
from torch.autograd import Variable
from dataLoader import getDataLoader
import models




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
    if config.dataset == 'Coco':
        config.data_path = './data/coco'
        config.num_classes = 80
        trainLoader, testLoader = getDataLoader(config)
        print('train samples num: ', len(trainLoader), '  test samples num: ', len(testLoader))
    elif config.dataset == 'VOC':
        config.data_path = './data/voc07'
        config.num_classes = 20
    else:
        print('Only support Coco and VOC!!')
        return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image-size', type=int,      default=32)
    parser.add_argument('--n-epochs',   type=int,      default=50)
    parser.add_argument('--batch-size', type=int,      default=128)
    parser.add_argument('--n-workers',  type=int,      default=4)
    parser.add_argument('--lr',         type=float,    default=0.1)
    parser.add_argument('--out-path',   type=str,      default='./output')
    parser.add_argument('--seed',       type=int,      default=0,           help='random seed for all')
    parser.add_argument('--log-step',   type=int,      default=100)
    parser.add_argument('--use-cuda',   type=bool,     default=True,        help='enables cuda')

    parser.add_argument('--dataset',    type=str,      default='Coco',      help='Coco or VOC')
    parser.add_argument('--mode',       type=str,      default='train',     help='train, test')
    parser.add_argument('--model',      type=str,      default='VGG',     help='model')
    parser.add_argument('--pretrained', type=str,      default='',          help='model for test or retrain')

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