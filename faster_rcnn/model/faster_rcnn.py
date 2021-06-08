 
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from backbone import ResNet50

class fasterRcnn(nn.Module):
    def __init__(self, classes, backbone):
        super(fasterRcnn, self).__init__()
        self.classes = classes
        self.num_class = len(classes)
        self.base_cnn = backbone

    def forward(self, input_batch):
        feature_maps = self.base_cnn(input_batch)
        print(feature_maps.shape)

def test():
    classes = ["Human", "dog", "cat"]
    resnet = ResNet50()

    test_fasterRcnn = fasterRcnn(classes=classes, backbone=resnet)
    images = list()
    labels = list()

    images.append(torch.rand([3, 370, 414]))
    images.append(torch.rand([3, 603, 752]))
    images.append(torch.rand([3, 423, 393]))
    images.append(torch.rand([3, 1157, 679]))
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

    label = {
        "boxes": torch.tensor([[180.2600,  31.0200, 435.1400, 355.1400],
        [236.0400, 155.8100, 402.4400, 351.0600]]),
        "labels": torch.tensor([64, 86]),
        "image_id": torch.tensor([30]),
        "area": torch.tensor([47675.6641, 16202.7979]),
        "iscrowd": torch.tensor([0, 0, ])
    }
    labels.append(label)

    label = {
        "boxes": torch.tensor([[234.9600,  22.0600, 668.0000, 401.2100]]),
        "labels": torch.tensor([24]),
        "image_id": torch.tensor([34]),
        "area": torch.tensor([92920.1562]),
        "iscrowd": torch.tensor([0, ])
    }
    labels.append(label)
    images = tuple(images)
    labels = tuple(labels)
    output = test_fasterRcnn(images)

test()