from coco_dataset import CocoDetection
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms as T
from coco_utils import get_coco
import presets 
import utils


def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 91),
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train, data_augmentation):
    return presets.DetectionPresetTrain(data_augmentation) if train else presets.DetectionPresetEval()

data_path = "../data/coco_2017/"
dataset, num_classes = get_dataset("coco", "train", get_transform(True, "ssd"),
                                       data_path)

train_loader = DataLoader(dataset=dataset, batch_size=4, collate_fn=utils.collate_fn)
print("number of sample: ", dataset.__len__())

print("import some sample")
for idx, sample in enumerate(train_loader):
    images, labels = sample
    for image, label in zip(images, labels):
        for k, v in label.items():
            if k=="masks":
                continue
            print(k, v)
            print()
        print(2*"\n")
    if idx == 0:
        break