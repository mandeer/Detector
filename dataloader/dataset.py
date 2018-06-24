# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
from PIL import Image


class Label(object):
    def __init__(self):
        self.imgName = ''
        self.bboxes = []
        self.labels = []


class DataSet(data.Dataset):
    ''' Load image, labels, boxes from a list file.
        The list file is like:
        a.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
    '''
    def __init__(self, root, list_file, transform=None):
        '''
        Args:
          root:         (str) ditectory to images.
          list_file:    (str/[str]) path to index file.
          transform:    (function) image/box transforms.
        '''
        self.root = root
        self.transform = transform

        self.dataes = []

        if isinstance(list_file, list):
            # Cat multiple list files together.
            # This is especially useful for voc07/voc12 combination.
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file

        with open(list_file) as file:
            lines = file.readlines()
            self.num_imgs = len(lines)

        for line in lines:
            data = Label()
            splited = line.strip().split()
            data.imgName = splited[0]
            num_boxes = (len(splited) - 1) // 5
            for i in range(num_boxes):
                xmin = splited[1+5*i]
                ymin = splited[2+5*i]
                xmax = splited[3+5*i]
                ymax = splited[4+5*i]
                c = splited[5+5*i]
                data.bboxes.append([float(xmin),float(ymin),float(xmax),float(ymax)])
                data.labels.append(int(c))
            self.dataes.append(data)

    def __getitem__(self, idx):
        ''' Load image.

        Args:
          idx: (int) image index.

        Returns:
          img:      (tensor) image tensor.
          boxes:    (tensor) bounding box targets.
          labels:   (tensor) class label targets.
        '''
        # Load image and boxes.
        data = self.dataes[idx]
        img = Image.open(os.path.join(self.root, data.imgName)).convert('RGB')

        boxes = torch.from_numpy(np.array(data.bboxes, dtype=np.float32))
        labels = torch.from_numpy(np.array(data.labels, dtype=np.int64))
        if self.transform:
            img, boxes, labels = self.transform(img, boxes, labels)
        return img, boxes, labels

    def __len__(self):
        return self.num_imgs


if __name__ == '__main__':
    import cv2
    from PIL import ImageDraw
    root = '../datasets/voc/VOC2012/JPEGImages'
    list_file = '../datasets/voc/voc12_trainval.txt'
    dataset = DataSet(root, list_file)

    num = len(dataset)
    print('num: ', num)
    for i in range(num):
        img, boxes, labels = dataset[i]

        # image = transforms.ToPILImage(img)
        image = img
        imageDraw = ImageDraw.Draw(image)
        num_obj, _ = boxes.shape
        for j in range(num_obj):
            imageDraw.rectangle([boxes[j][0], boxes[j][1], boxes[j][2], boxes[j][3]], outline='red')
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        cv2.imshow("OpenCV", image)
        cv2.waitKey(1000)