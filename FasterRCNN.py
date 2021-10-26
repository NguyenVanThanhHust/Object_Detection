# Learn from youtube + notebook
import os.path as osp
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from torchvision.models.resnet import resnet50
from collections import OrderedDict

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(0))
else:
    print("There is no availble gpu")

image = cv2.imread("291.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

to_vis = False

# Create fake bounding boxes
bboxes = np.array([
    [160, 180, 180, 200], 
    [184, 126, 245, 342]
])

labels = np.array([1, 2])

im_height, im_width, im_chanel = image.shape

# Resize input images to (h=800, w=800)
image = cv2.resize(image, dsize=(800, 800))

# Resize box according to images
w_ratio, h_ratio = 800/im_width, 800/im_height
bboxes[:, 0] = bboxes[:, 0]/w_ratio
bboxes[:, 1] = bboxes[:, 1]/h_ratio
bboxes[:, 2] = bboxes[:, 2]/w_ratio
bboxes[:, 3] = bboxes[:, 3]/h_ratio

class Resnet50FPN(nn.Module):
    def __init__(self):
        super(Resnet50FPN, self).__init__()
        if osp.isfile("./pretrain_models/resnet50-19c8e357.pth"):
            self.resnet = torchvision.models.resnet50()
            self.resnet.load_state_dict(torch.load("./pretrain_models/resnet50-19c8e357.pth"))
        else:
            self.resnet = torchvision.models.resnet50(pretrained=False)
        children = list(self.resnet.children())
        self.backbone = nn.Sequential(*children[:6])

    def forward(self, x):
        feat = self.backbone(x)
        return feat

backbone_model = Resnet50FPN()
backbone_model.to(device)

transform = transforms.Compose([transforms.ToTensor()])
input_tensor = transform(image).to(device)
input_tensor = input_tensor.unsqueeze(dim=0)
output_feat = backbone_model(input_tensor)
print(output_feat.shape)

feat_size = 800/16
center_x = np.arange(16, (feat_size+1)*16, 16) 
center_y = np.arange(16, (feat_size+1)*16, 16)

center_anchors = np.zeros((2500, 2))
index = 0
for x in range(len(center_x)):
    for y in range(len(center_y)):
        center_anchors[index, 1] = center_x[x] - 8
        center_anchors[index, 0] = center_y[y] - 8
        index +=1
if to_vis:
    img_clone = np.copy(image)
    for i in range(center_anchors.shape[0]):
        img_clone = cv2.circle(img_clone, (int(center_anchors[i][0]), int(center_anchors[i][1])), radius=1, color=(255, 255, 0), thickness=1)
    cv2.imwrite("anchor.jpg", img_clone)

# Generate 9 anchor boxes for each anchor in 2500 anchors
ratios = [0.5, 1, 2]
scales = [8, 16, 32]
sub_sample = 16
anchor_boxes = np.zeros((int(feat_size*feat_size*9), 4))
index = 0
for center_anchor in center_anchors:
    center_x, center_y = center_anchor
    for i in range(len(ratios)):
        for j in range(len(scales)):
            h = sub_sample * scales[j] * np.sqrt(ratios[i])
            w = sub_sample * scales[j] * np.sqrt(1./ ratios[i])
            anchor_boxes[index, 0] = center_y - h / 2.
            anchor_boxes[index, 1] = center_x - w / 2.
            anchor_boxes[index, 2] = center_y + h / 2.
            anchor_boxes[index, 3] = center_x + w / 2.
            index += 1

if to_vis:
    img_clone = np.copy(image)
    for i in range(11025, 11034):
        x0 = int(anchor_boxes[i][1])
        y0 = int(anchor_boxes[i][0])
        x1 = int(anchor_boxes[i][3])
        y1 = int(anchor_boxes[i][2])
        cv2.rectangle(img_clone, (x0, y0), (x1, y1), color=(255, 255, 2550), thickness=3) 
    cv2.imwrite("box.jpg", img_clone)

# Remove anchor box which have coord excess image
index_inside = np.where(
    (anchor_boxes[:, 0] >= 0) &
    (anchor_boxes[:, 1] >= 0) &
    (anchor_boxes[:, 2] <= 800) &
    (anchor_boxes[:, 3] <= 800)
)[0]

valide_anchor_boxes = anchor_boxes[index_inside]

import pdb; pdb.set_trace()