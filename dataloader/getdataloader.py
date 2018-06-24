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

    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=4)
    testLoader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=4)

    return trainLoader, testLoader


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='getDataLoader test')

    config = parser.parse_args()
    config.train_root = '../datasets/voc/VOC2007/JPEGImages'
    config.train_label_file = '../datasets/voc/voc07_trainval.txt'
    config.test_root = '../datasets/voc/VOC2007/JPEGImages'
    config.test_label_file = '../datasets/voc//voc07_test.txt'
    config.batch_size = 1

    args = vars(config)
    print('------------ Options -------------')
    for key, value in sorted(args.items()):
        print('%16.16s: %16.16s' % (str(key), str(value)))
    print('-------------- End ----------------')

    trainLoader, testLoader = get_data_loader(config)
    print('train samples num: ', len(trainLoader), '  test samples num: ', len(testLoader))

    for ii, (img, boxes, labels) in enumerate(trainLoader):
        print(img.shape)
        print(boxes.shape)
        print(labels.shape)
        print(ii)
