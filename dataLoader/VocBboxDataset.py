# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils import data
from torchvision import transforms


class VocBboxDataset(data.Dataset):
    def __init__(self, data_dir, split='trainval', transform=None, resizeImage=True):

        if split not in ['train', 'trainval', 'val']:
            if not (split == 'test' and data_dir.split('/')[-1] == 'VOC2007'):
                print(
                    'please pick split from \'train\', \'trainval\', \'val\''
                    'for 2012 dataset. For 2007 dataset, you can pick \'test\''
                    ' in addition to the above mentioned splits.'
                )

        id_list_file = os.path.join(data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.label_names = VOC_BBOX_LABEL_NAMES
        self.resizeImage = resizeImage
        if transform is None:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # This makes it into [0,1]
                normalize
            ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        anno = ET.parse(os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        for obj in anno.findall('object'):
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([int(bndbox_anno.find(tag).text) - 1
                         for tag in ('xmin', 'ymin', 'xmax', 'ymax')])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)

        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = Image.open(img_file).convert('RGB')
        W, H = img.size
        bbox /= [W, H, W, H]
        if self.resizeImage:
            img = preprocess(img)
        img = self.transform(img)

        return img, bbox, label

    def __len__(self):
        return len(self.ids)


def preprocess(img, min_size=600, max_size=1000):
    W, H = img.size
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img.resize((int(W * scale), int(H * scale)), Image.ANTIALIAS)
    return img


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
)
