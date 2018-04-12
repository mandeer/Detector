# -*- coding: utf-8 -*-

from .VocBboxDataset import VocBboxDataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

def getVocDataLoader(config):
    assert config.dataset == 'VOC07' or config.dataset == 'VOC12'
    if config.dataset == 'VOC07':
        train_data = VocBboxDataset(config.data_path, split='trainval')
        test_data = VocBboxDataset(config.data_path, split='test')
    else:
        train_data = VocBboxDataset(config.data_path, split='train')
        test_data = VocBboxDataset(config.data_path, split='val')

    img, bbox, label = train_data[0]
    print(img.size)
    print(bbox)
    print(label)