import sys
import math 
import random
from typing import Tuple, List, Dict
from torch import nn, Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torch.autograd import Variable
import numpy as np

from model.resnet import resnet152, resnet101, resnet50, resnet34, resnet18
from faster_rcnn.model.faster_rcnn_base import fasterRcnn
from .vgg import VGG

torch.cuda.empty_cache()

class faster_rcnn_resnet(fasterRcnn):
    def __init__(self, classes=None, backbone=None):
        super(faster_rcnn_resnet, self).__init__()
        if backbone=="resnet_18":
            self.backbone = resnet18()
        else:
            raise NotImplementedError

        self.classes = classes
        # self.num_class = len(classes)
        self.num_class = 3
        self.base_cnn = backbone
    #   Resize(min_size=(800,), max_size=1333, mode='bilinear')

        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.size_divisible = 32

    def normalize(self, image):
        if not image.is_floating_point():
            raise TypeError(
                f"Expected input images to be of floating type (in range [0, 1]), "
                f"but found type {image.dtype} instead"
            )
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def max_by_axis(self, list_size):
        """
        Input list of size(C, H, W)
        list_size = [[C_1, H_1, W_1], [C_2, H_2, W_2],...]
        """

        shape_list = np.array(list_size)
        assert shape_list.shape[1] == 3, "check shape"
        max_c, max_h, max_w = shape_list.max(axis=0)
        return [max_c, max_h, max_w]

    def batch_images(self, images, size_divisible=32):
        if torchvision._is_tracing():
            print("not implemented yet")
            sys.exit()
        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    def transform(self, images, targets=None):
        """
        This function convert list of images and list of targets dict into new form
        """
        assert isinstance(images, list), "images must be list of tensor"
        # convert targets as in https://github.com/pytorch/vision/blob/e4c56081ded40403bc3b53ff73ed819d6a46f33e/torchvision/models/detection/transform.py#L65
        if targets:
            targets_copy: List[Dict[str, Tensor]] = []
            for t in targets:
                data: Dict[str, Tensor] = {}
                for k, v in t.items():
                    data[k] = v
                targets_copy.append(data)
            targets = targets_copy
        for i in range(len(images)):
            image = images[i]
            target = targets[i] if targets is not None else None
            assert image.dim() == 3, "images must be list of tensor of shape [C, H, W] got {}".format(image.shape)
            image = self.normalize(image)
            images[i] =image
            if targets is not None and target is not None:
                targets[i] = target
        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images, size_divisible=self.size_divisible)
        image_sizes_list: List[Tuple[int, int]] = []
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        return images, image_sizes, targets

    def forward(self, images, targets=None):
        images, image_sizes, targets = self.transform(images, targets)
        print(images.shape)
        feature_maps = self.base_cnn(images)
        print(feature_maps.shape)

def test():
    classes = ["Human", "dog", "cat"]

    test_fasterRcnn = faster_rcnn_resnet(classes=classes, backbone="resnet_18")
    images = list()
    labels = list()
    images.append(torch.rand([3, 370, 414]))
    images.append(torch.rand([3, 603, 752]))
    # images.append(torch.rand([3, 423, 393]))
    # images.append(torch.rand([3, 1157, 679]))
    label = {
        "boxes": torch.tensor([[  0.0000, 176.6900, 246.0000, 451.0000],
        [ 41.6000, 218.2700, 246.0000, 451.0000],
        [  0.0000,   2.5100, 226.4800, 377.6300],
        [168.2000,  29.3600, 243.7500,  75.8900],
        [177.7000,  62.6600, 246.0000, 133.1700],
        [156.0500,   0.0000, 246.0000,  62.5600]]),
        "labels": torch.tensor([51, 56, 51, 55, 55, 55]),
        "image_id": torch.tensor([9]),
        "area": torch.tensor([120057.1406,  44434.7500,  49577.9453,  24292.7812,   2239.2925,
                1658.8914,   3609.3030,   2975.2759]),
        "iscrowd": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
    }
    labels.append(label)

    label = {
        "boxes": torch.tensor([[ 13.5000,  13.0300, 228.4700, 310.1900],
        [428.9600, 309.4900, 560.9900, 351.0000]]),
        "labels": torch.tensor([25, 25]),
        "image_id": torch.tensor([25]),
        "area": torch.tensor([19686.5977,  2785.8477]),
        "iscrowd": torch.tensor([0, 0, ])
    }
    labels.append(label)

    # label = {
    #     "boxes": torch.tensor([[180.2600,  31.0200, 435.1400, 355.1400],
    #     [236.0400, 155.8100, 402.4400, 351.0600]]),
    #     "labels": torch.tensor([64, 86]),
    #     "image_id": torch.tensor([30]),
    #     "area": torch.tensor([47675.6641, 16202.7979]),
    #     "iscrowd": torch.tensor([0, 0, ])
    # }
    # labels.append(label)

    # label = {
    #     "boxes": torch.tensor([[234.9600,  22.0600, 668.0000, 401.2100]]),
    #     "labels": torch.tensor([24]),
    #     "image_id": torch.tensor([34]),
    #     "area": torch.tensor([92920.1562]),
    #     "iscrowd": torch.tensor([0, ])
    # }
    # labels.append(label)
    
    images = tuple(images)
    targets = tuple(labels)
    # device = torch.device("cpu")
    device = torch.device("cuda")
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
    test_fasterRcnn.to(device)
    output = test_fasterRcnn(images, targets)

test()