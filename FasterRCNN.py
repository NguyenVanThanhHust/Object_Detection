# Learn from youtube + notebook
# https://fractaldle.medium.com/guide-to-build-faster-rcnn-in-pytorch-95b10c273439
import os.path as osp
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

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
valid_index = np.where(
    (anchor_boxes[:, 0] >= 0) &
    (anchor_boxes[:, 1] >= 0) &
    (anchor_boxes[:, 2] <= 800) &
    (anchor_boxes[:, 3] <= 800)
)[0]

valid_anchor_boxes = anchor_boxes[valid_index]
# Find positve and negative anchor boxes for training process
# Each anchor box have iou with gts > pos_iou_thresh is valid
# Each anchor box have iou with gts < neg_iou_thresh is invalid
# The rest doesn't contribute to training process
pos_iou_thresh = 0.7 
neg_iou_thresh = 0.3
all_ious = []
for bbox in bboxes:
    x1, y1, x2, y2 = bbox
    xmin = np.maximum(x1, valid_anchor_boxes[:, 0]) 
    ymin = np.maximum(y1, valid_anchor_boxes[:, 1]) 
    xmax = np.minimum(x2, valid_anchor_boxes[:, 2]) 
    ymax = np.minimum(y2, valid_anchor_boxes[:, 3])
    widths = np.maximum(xmax - xmin + 1.0, 0.0)
    heights = np.maximum(ymax - ymin + 1.0, 0.0)
    inters = widths*heights 
    unions = (
        (xmax - xmin + 1.0) * (ymax - ymin + 1.0) +
        (x2 - x1 + 1.0) * (y2 -y1 + 1.0) -
        inters
    )
    ious = inters / unions
    all_ious.append(ious)

all_ious = np.array(all_ious)
print(all_ious.shape)

# We need to assign positive anchors with highest iou with ground truth box
# or anchor have iou > pos_iou_thresh, we follow below step

# Create labels for anchor boxes
anchor_box_labels = np.zeros(len(valid_index), dtype=np.int32) - 1
# Find max iou of each anchor box
max_ious = all_ious.argmax(axis=0)
# Assign as above
anchor_box_labels[max_ious > pos_iou_thresh] = 1
anchor_box_labels[max_ious < neg_iou_thresh] = 0

# If all the ious is < 0.7, then assign box with maximum iou with gt as true
index_max_ious = all_ious.argmax(1) 
anchor_box_labels[index_max_ious] = 1

# The Faster_R-CNN paper phrases as follows Each mini-batch arises from a single image that contains
# many positive and negitive example anchors, but this will bias 
# towards negitive samples as they are dominate. Instead, we randomly
# sample 256 anchors in an image to compute the loss function of a mini-batch,
#  where the sampled positive and negative anchors have a ratio of up to 1:1. 
#  If there are fewer than 128 positive samples in an image, 
#  we pad the mini-batch with negitive ones

pos_ratio = 0.5
n_sample = 256
n_pos = int(pos_ratio * n_sample)
n_neg = n_sample - n_pos

pos_ids = np.where(anchor_box_labels == 1)[0] # return tuple
if len(pos_ids) > n_pos:
    disable_ids = np.random.choice(pos_ids, size=(len(pos_ids) - n_pos), replace=False)
    anchor_box_labels[disable_ids] = -1

neg_ids = np.where(anchor_box_labels == 0)[0] # return tuple
if len(neg_ids) > n_neg:
    disable_ids = np.random.choice(neg_ids, size=(len(neg_ids) - n_neg), replace=False)
    anchor_box_labels[disable_ids] = -1

# For each anchor box, find location of corresponding ground truths
# Find which ground truth box is associated with anchor box
argmax_ious = all_ious.argmax(axis=0)
max_iou_boxes = bboxes[argmax_ious]

height = valid_anchor_boxes[:, 2] - valid_anchor_boxes[:, 0]
width = valid_anchor_boxes[:, 3] - valid_anchor_boxes[:, 1]
center_y = valid_anchor_boxes[:, 0] + 0.5 * height
center_x = valid_anchor_boxes[:, 1] + 0.5 * width

base_height = max_iou_boxes[:, 2] - max_iou_boxes[:, 0]
base_width = max_iou_boxes[:, 3] - max_iou_boxes[:, 1]
base_center_y = max_iou_boxes[:, 0] + 0.5 * base_height
base_center_x = max_iou_boxes[:, 1] + 0.5 * base_width

eps = np.finfo(height.dtype).eps
height = np.maximum(height, eps)
width = np.maximum(width, eps)
dy = (base_center_y - center_y) / height
dx = (base_center_x - center_x) / width
dh = np.log(base_height / height)
dw = np.log(base_width / width)
anchor_locs = np.vstack((dy, dx, dh, dw)).transpose()

anchor_labels = np.empty((len(anchor_boxes), ), dtype=anchor_box_labels.dtype)
anchor_labels.fill(-1)
anchor_labels[valid_index] = anchor_box_labels

anchor_locations = np.empty(shape=(len(anchor_boxes), anchor_boxes.shape[1]), dtype=anchor_locs.dtype)
anchor_locations.fill(0)
anchor_locations[valid_index, :] = anchor_locs

import pdb; pdb.set_trace()

# Define region proposal network