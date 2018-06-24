# -*- coding: utf-8 -*-

import torchvision.transforms as transforms



def transform_train(img, boxes, labels):
    img, boxes = random_flip(img, boxes)
    img, boxes = resize(img, boxes, size=img_size, max_size=img_size)
    img = pad(img, (img_size, img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels


def transform_test(img, boxes, labels):
    img, boxes = resize(img, boxes, size=img_size, max_size=img_size)
    img = pad(img, (img_size, img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels