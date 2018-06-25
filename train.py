# -*- coding: utf-8 -*-

import os
import random
import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
from dataloader import get_data_loader
from models import RetinaNet
from models.loss import FocalLoss

class Solver(object):
    def __init__(self, config, model, trainLoader, testLoader):
        self.model = model
        self.trainLoader = trainLoader
        self.testLoader = testLoader
        self.n_classes = config.n_classes
        self.use_cuda = config.use_cuda

        self.optimizer = optim.SGD(self.model.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-4)
        self.criterion = FocalLoss(num_classes=self.n_classes)
        if self.use_cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        self.n_epochs = config.n_epochs
        self.log_step = config.log_step
        self.out_path = config.out_path

    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.model.train()
        train_loss = 0
        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(self.trainLoader):
            if self.use_cuda:
                inputs = Variable(inputs).cuda()
                loc_targets = Variable(loc_targets).cuda()
                cls_targets = Variable(cls_targets).cuda()


            self.optimizer.zero_grad()
            loc_preds, cls_preds = self.model(inputs)
            loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss.backward()
            self.optimizer.step()

            train_loss += float(loss.data[0])
            print('train_loss: %.3f | avg_loss: %.3f [%d/%d]'
                  % (loss.data[0], train_loss / (batch_idx + 1), batch_idx + 1, len(self.trainLoader)))

    def test(self, epoch):
        print('\nTest')
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(self.testLoader):
                if self.use_cuda:
                    inputs = Variable(inputs).cuda()
                    loc_targets = Variable(loc_targets).cuda()
                    cls_targets = Variable(cls_targets).cuda()

                loc_preds, cls_preds = self.model(inputs)
                loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
                test_loss += float(loss.data[0])
                print('test_loss: %.3f | avg_loss: %.3f [%d/%d]'
                      % (loss.data[0], test_loss / (batch_idx + 1), batch_idx + 1, len(self.testLoader)))

        # Save checkpoint
        global best_loss
        test_loss /= len(self.testLoader)
        if test_loss < best_loss:
            print('Saving..')
            state = {
                'net': self.model.state_dict(),
                'loss': test_loss,
                'epoch': epoch,
            }
            if not os.path.isdir(os.path.dirname(args.checkpoint)):
                os.mkdir(os.path.dirname(args.checkpoint))
            torch.save(state, args.checkpoint)
            best_loss = test_loss

def main(config):
    # use cuda ?
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

    # dataLoader
    trainLoader, testLoader = get_data_loader(config)
    print('train samples num: ', len(trainLoader), '  test samples num: ', len(testLoader))

    # net
    model = RetinaNet(num_classes=config.n_classes)
    print(model)

    solver = Solver(config, model, trainLoader, testLoader)
    for epoch in range(config.n_epochs):
        solver.train(epoch)
        solver.test(epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image-size',         type=int,       default=2)
    parser.add_argument('--n-epochs',           type=int,       default=50)
    parser.add_argument('--batch-size',         type=int,       default=2)
    parser.add_argument('--n-workers',          type=int,       default=4)
    parser.add_argument('--lr',                 type=float,     default=0.01)
    parser.add_argument('--out-path',           type=str,       default='./output')
    parser.add_argument('--seed',               type=int,       default=666,            help='random seed for all')
    parser.add_argument('--log-step',           type=int,       default=100)
    parser.add_argument('--use-cuda',           type=bool,      default=True,           help='enables cuda')


    parser.add_argument('--train-root',         type=str,       default='./datasets/voc/VOC2007/JPEGImages')
    parser.add_argument('--train-label-file',   type=str,       default='./datasets/voc/voc07_trainval.txt')
    parser.add_argument('--test-root',          type=str,       default='./datasets/voc/VOC2007/JPEGImages')
    parser.add_argument('--test-label-file',    type=str,       default='./datasets/voc//voc07_test.txt')
    parser.add_argument('--n_classes',          type=int,       default=20)
    parser.add_argument('--mode',               type=str,       default='train',        help='train, test')
    parser.add_argument('--model',              type=str,       default='RetinaNet',    help='model')
    parser.add_argument('--pretrained',         type=str,       default='',             help='model for test or retrain')

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