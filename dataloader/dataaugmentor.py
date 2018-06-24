# -*- coding: utf-8 -*-

import random
from PIL import Image
import torch


class DataAugmentor(object):
    def __init__(self, imgSize=640):
        if isinstance(imgSize, (int, float)):
            self.imgW = imgSize
            self.imgH = imgSize
        else:
            self.imgW = imgSize[0]
            self.imgH = imgSize[1]

    def pad(self, img, target_size):
        '''Pad image with zeros to the specified size.

        Args:
          img: (PIL.Image) image to be padded.
          target_size: (tuple) target size of (ow,oh).

        Returns:
          img: (PIL.Image) padded image.

        Reference:
          `tf.image.pad_to_bounding_box`
        '''
        w, h = img.size
        canvas = Image.new('RGB', target_size)
        canvas.paste(img, (0, 0))  # paste on the left-up corner
        return canvas

    def random_flip(self, img, boxes):
        '''Randomly flip PIL image.

        If boxes is not None, flip boxes accordingly.

        Args:
          img: (PIL.Image) image to be flipped.
          boxes: (tensor) object boxes, sized [#obj,4].

        Returns:
          img: (PIL.Image) randomly flipped image.
          boxes: (tensor) randomly flipped boxes.
        '''
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w = img.width
            if boxes is not None:
                xmin = w - boxes[:, 2]
                xmax = w - boxes[:, 0]
                boxes[:, 0] = xmin
                boxes[:, 2] = xmax
        return img, boxes

    def resize(self, img, boxes, size, max_size=1000, random_interpolation=False):
        '''Resize the input PIL image to given size.

        If boxes is not None, resize boxes accordingly.

        Args:
          img: (PIL.Image) image to be resized.
          boxes: (tensor) object boxes, sized [#obj,4].
          size: (tuple or int)
            - if is tuple, resize image to the size.
            - if is int, resize the shorter side to the size while maintaining the aspect ratio.
          max_size: (int) when size is int, limit the image longer size to max_size.
                    This is essential to limit the usage of GPU memory.
          random_interpolation: (bool) randomly choose a resize interpolation method.

        Returns:
          img: (PIL.Image) resized image.
          boxes: (tensor) resized boxes.

        Example:
        >> img, boxes = resize(img, boxes, 600)  # resize shorter side to 600
        >> img, boxes = resize(img, boxes, (500,600))  # resize image size to (500,600)
        >> img, _ = resize(img, None, (500,600))  # resize image only
        '''
        w, h = img.size
        if isinstance(size, int):
            size_min = min(w, h)
            size_max = max(w, h)
            sw = sh = float(size) / size_min
            if sw * size_max > max_size:
                sw = sh = float(max_size) / size_max
            ow = int(w * sw + 0.5)
            oh = int(h * sh + 0.5)
        else:
            ow, oh = size
            sw = float(ow) / w
            sh = float(oh) / h

        method = random.choice([
            Image.BOX,
            Image.NEAREST,
            Image.HAMMING,
            Image.BICUBIC,
            Image.LANCZOS,
            Image.BILINEAR]) if random_interpolation else Image.BILINEAR
        img = img.resize((ow, oh), method)
        if boxes is not None:
            boxes = boxes * torch.FloatTensor([sw, sh, sw, sh])
        return img, boxes
