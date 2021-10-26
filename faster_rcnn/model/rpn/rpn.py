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
import numpy as np
import os
from easydict import EasyDict as edict
try:
    from .proposal_layer import _ProposalLayer
except :
    from proposal_layer import _ProposalLayer

class config:
    def __init__(self, ):
        self.ANCHOR_SCALES = [8, 16, 32]
        self.ANCHOR_RATIOS = [0.5, 1.0, 2.0]
        self.FEAT_STRIDE = [16, ]

# nc is number classes
class RegionProposalLayer(nn.Module):
    def __init__(self, cfg, input_depth=512):
        super(RegionProposalLayer, self).__init__()
        self.cfg = cfg
        self.input_depth = input_depth

        # Define layer in here
        self.RPN_Conv = nn.Conv2d(in_channels=self.input_depth,
                                    out_channels=512, 
                                    kernel_size=3, 
                                    stride=1,
                                    padding=1,
                                    bias=False)

        # define bg/fg classifcation score layer
        self.nc_score_out = len(self.cfg.ANCHOR_SCALES) * len(self.cfg.ANCHOR_RATIOS) * 2 # 2(bg/fg) * 9 
        self.RPN_cls_score = nn.Conv2d(in_channels=512,
                                        out_channels=self.nc_score_out,
                                        kernel_size=1, 
                                        stride=1, 
                                        padding=0)

        # define bbox offset layer
        self.nc_bbox_out = len(self.cfg.ANCHOR_SCALES) * len(self.cfg.ANCHOR_RATIOS) * 4 # 2(bg/fg) * 9 
        self.RPN_bbox_pred = nn.Conv2d(in_channels=512,
                                        out_channels=self.nc_bbox_out,
                                        kernel_size=1, 
                                        stride=1, 
                                        padding=0)


        self.RPN_proposal = _ProposalLayer(cfg.FEAT_STRIDE, cfg.ANCHOR_SCALES, cfg.ANCHOR_RATIOS)


    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, features):
        """
        features: torch.cuda.Tensor, shape should be (batch_size, 512, 7, 7)
        """
        rpn_features = F.relu(self.RPN_Conv(features), inplace=True)
        
        rpn_cls_scores = self.RPN_cls_score(rpn_features)
        rpn_cls_scores_reshape = self.reshape(x=rpn_cls_scores, d=2)
        rpn_cls_probs_reshape = F.softmax(rpn_cls_scores_reshape, 1)
        rpn_cls_probs = self.reshape(rpn_cls_probs_reshape, self.nc_score_out)

        # This is rpn offset to the anchor boxes
        rpn_pred_bboxes = self.RPN_bbox_pred(rpn_features)

        # get regions of interest
        image_shape = []
        cfg_key = "TRAIN"
        rois = self.RPN_proposal(rpn_cls_probs, rpn_pred_bboxes, image_shape, cfg_key)
        
        print("rpn_features.shape", rpn_features.shape)
        print("rpn_bboxes.shape", rpn_pred_bboxes.shape)
        print("rpn_cls_probs.shape", rpn_cls_probs.shape)
        # scores.shape torch.Size([4, 9, 7, 7])
        # rpn_features.shape torch.Size([4, 512, 7, 7])
        # rpn_bboxes.shape torch.Size([4, 36, 7, 7])

        # print(self.cfg)
        return rpn_features, rpn_cls_scores, rpn_pred_bboxes



def test():
    cfg = config()
    rpn_layer = RegionProposalLayer(cfg, input_depth=512)
    fake_features = torch.rand([4, 512, 7, 7])
    device = torch.device("cuda")
    fake_features = fake_features.to(device)
    rpn_layer = rpn_layer.to(device)
    results = rpn_layer(fake_features)
    rpn_features, rpn_cls_scores, rpn_bboxes = results
    print("rpn_bboxes.shape", rpn_bboxes.shape)

test()
