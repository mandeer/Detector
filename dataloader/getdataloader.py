# -*- coding: utf-8 -*-

import torch
import torchvision.transforms as transforms

from dataloader.dataset import DataSet
from dataloader.en_decoder import RetinaBoxCoder


box_coder = RetinaBoxCoder()


def transform_train(img, boxes, labels):
    # img, boxes = random_flip(img, boxes)
    # img, boxes = resize(img, boxes, size=img_size, max_size=img_size)
    # img = pad(img, (img_size, img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels


def transform_test(img, boxes, labels):
    # img, boxes = resize(img, boxes, size=img_size, max_size=img_size)
    # img = pad(img, (img_size, img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels


def get_data_loader(conf):
    trainset = DataSet(root=conf.train_root,
                       list_file=conf.train_label_file,
                       transform=transform_train)

    testset = DataSet(root=conf.test_root,
                      list_file=conf.test_label_file,
                      transform=transform_test)

    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)
    testLoader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=4)

    return trainLoader, testLoader


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='getDataLoader test')
    parser.add_argument('train-root',       default='../datasets/voc/VOC2007/JPEGImages',   type=str)
    parser.add_argument('train-label-file', default='../datasets/voc/voc07_trainval.txt',   type=str)
    parser.add_argument('test-root',        default='../datasets/voc/VOC2007/JPEGImages',   type=str)
    parser.add_argument('test-label-file',  default='../datasets/voc//voc07_test.txt',      type=str)

    config = parser.parse_args()
    args = vars(config)
    print('------------ Options -------------')
    for key, value in sorted(args.items()):
        print('%16.16s: %16.16s' % (str(key), str(value)))
    print('-------------- End ----------------')

    trainLoader, testLoader = get_data_loader(config)
    print('train samples num: ', len(trainLoader), '  test samples num: ', len(testLoader))