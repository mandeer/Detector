# -*- coding: utf-8 -*-

''' Microsoft COCO 数据集
http://images.cocodataset.org/zips/train2017.zip
http://images.cocodataset.org/annotations/annotations_trainval2017.zip

http://images.cocodataset.org/zips/val2017.zip
http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip

http://images.cocodataset.org/zips/test2017.zip
http://images.cocodataset.org/annotations/image_info_test2017.zip
'''

import torch
from torchvision import datasets
from torchvision import transforms


def getDataLoader(config):
    assert config.dataset == 'Coco'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),  # This makes it into [0,1]
        normalize
    ])

    train_data = datasets.CocoDetection(root=config.data_path + '/train2017',
                                        annFile=config.data_path + '/annotations/instances_train2017.json',
                                        transform=transform)

    val_data = datasets.CocoDetection(root=config.data_path + '/val2017',
                                      annFile=config.data_path + '/annotations/instances_val2017.json',
                                      transform=transform)

    trainLoader = torch.utils.data.DataLoader(dataset=train_data,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.n_workers,
                                              drop_last=True)
    testLoader = torch.utils.data.DataLoader(dataset=val_data,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.n_workers,
                                             drop_last=True)

    return trainLoader, testLoader