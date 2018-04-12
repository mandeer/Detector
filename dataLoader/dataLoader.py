# -*- coding: utf-8 -*-

from .VocBboxDataset import VOC_BBOX_LABEL_NAMES
from .VocBboxDataset import VocBboxDataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

def getVocDataLoader(config):
    assert config.dataset == 'VOC07' or config.dataset == 'VOC12'

    transform = transforms.Compose([
        transforms.ToTensor(),  # This makes it into [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if config.dataset == 'VOC07':
        train_data = VocBboxDataset(config.data_path, split='trainval', transforms=transform)
        val_data = VocBboxDataset(config.data_path, split='test', transforms=transform)
    else:
        train_data = VocBboxDataset(config.data_path, split='train', transforms=transform)
        val_data = VocBboxDataset(config.data_path, split='val', transforms=transform)



    trainLoader = torch.utils.data.DataLoader(dataset=train_data,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.n_workers,
                                              drop_last=True)
    valLoader = torch.utils.data.DataLoader(dataset=val_data,
                                            batch_size=config.batch_size,
                                            shuffle=False,
                                            num_workers=config.n_workers,
                                            drop_last=True)

    return trainLoader, valLoader