import os
import sys
import collections
import torch
from torch.utils.data.dataloader import DataLoader, DataLoaderIter
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image

if sys.version_info[0] == 2:
    import Queue as queue
    string_classes = basestring
else:
    import queue
    string_classes = (str, bytes)


def getDataloaders(data, config_of_data, splits=['train', 'val'],
                   data_root='data', batch_size=16, normalized=True,
                   num_workers=3, **kwargs):
    train_loader, val_loader, test_loader = None, None, None

    if data.find('coco') >= 0:
        print('loading ' + data)
        print(config_of_data)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        common_trans = [transforms.ToTensor()]
        if normalized:
            print('dataset is normalized')
            common_trans.append(normalize)
        compose = transforms.Compose(common_trans)

        # uses last 5000 images of the original val split as the
        # mini validation set
        if 'train' in splits:
            train_set = CocoDetection(data_root, config_of_data['train_split'],
                                      scale_size=config_of_data['scale_size'],
                                      transform=compose)
            train_loader = DataLoader(
                train_set, batch_size=batch_size,
                collate_fn=coco_collate,
                shuffle=True,
                num_workers=num_workers, pin_memory=False)
        if 'val' in splits:
            val_set = CocoDetection(data_root, config_of_data['val_split'],
                                      scale_size=config_of_data['scale_size'],
                                      transform=compose)
            val_loader = DataLoader(
                val_set, batch_size=batch_size,
                collate_fn=coco_collate,
                shuffle=True,
                num_workers=num_workers, pin_memory=False)
        if 'test' in splits:
            test_set = CocoDetection(data_root, config_of_data['test_split'],
                                      scale_size=config_of_data['scale_size'],
                                      transform=compose)
            test_loader = DataLoader(
                test_set, batch_size=batch_size,
                collate_fn=coco_collate,
                shuffle=True,
                num_workers=num_workers, pin_memory=False)
    else:
        raise NotImplemented
    return train_loader, val_loader, test_loader


# Based on CocoDetection in torchvision
class CocoDetection(torch.utils.data.Dataset):

    def __init__(self, root, annfile, scale_size=None, transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.annfile = annfile
        self.coco = COCO(os.path.join(root, annfile))
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.scale_size = scale_size
        self.ord2cid = sorted(self.coco.cats.keys())
        self.cid2ord = {i: o for o, i in enumerate(self.ord2cid)}

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # search across both train2014 and val2014 in case of using trainval35k
        for subdir in ('train2014', 'val2014'):
            tmppath = os.path.join(self.root, subdir,
                                   self.coco.loadImgs(img_id)[0]['file_name'])
            if os.path.isfile(tmppath):
                path = tmppath

        # load image
        try:
            img = Image.open(path).convert('RGB')
        except:
            print('path:', path)
        for ann in anns:
            # COCO uses x, y, w, h, but Faster RCNN uses x1, y1, x2, y2
            ann['bbox'][2] += ann['bbox'][0]
            ann['bbox'][3] += ann['bbox'][1]
            # original id is in [1, 90] with skips, we convert them to a compact range [0, 79]
            ann['ordered_id'] = self.cid2ord[ann['category_id']]
            ann['scale_ratio'] = 1.
            # # get the mask for mask rcnn
            # ann['mask'] = torch.from_numpy(self.coco.annToMask(ann)).float().unsqueeze(0)

        # scaling image make shorter edge being scale_size 
        if self.scale_size is not None:
            w, h = img.size
            scale_ratio = self.scale_size / w if w < h else self.scale_size / h
            if scale_ratio != 1.:
                img = img.resize((int(w * scale_ratio), int(h * scale_ratio)),
                                 Image.BILINEAR)
                for ann in anns:
                    ann['area'] *= scale_ratio**2
                    ann['bbox'] = [x * scale_ratio for x in ann['bbox']]
                    # print(ann['segmentation'])
                    # ann['segmentation'] = [[x * scale_ratio for x in y]
                    #                        for y in ann['segmentation']]
                    # mask = transforms.ToPILImage()(ann['mask'])
                    # mask = mask.resize((round(w * scale_ratio),
                    #                     round(h * scale_ratio)),
                    #                    Image.BILINEAR)
                    # ann['mask'] = transforms.ToTensor()(mask)
                    ann['scale_ratio'] = scale_ratio
        
        # convert image to tensor and normalize it
        if self.transform is not None:
            img = self.transform(img)

        return img, anns

    def __len__(self):
        return len(self.ids)


def coco_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size, or put collade recursively for dict"
    if isinstance(batch[0], tuple):
        # if each batch element is not a tensor, then it should be a tuple
        # of tensors; in that case we collate each element in the tuple
        transposed = zip(*batch)
        return [coco_collate(samples) for samples in transposed]
    return batch
