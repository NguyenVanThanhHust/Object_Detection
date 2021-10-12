import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

try:
    from .generate_anchor import generate_anchors
    from .bbox_tranform import bbox_transform, bbox_transform_inv, clip_boxes
except:
    from generate_anchor import generate_anchors
    from bbox_tranform import bbox_transform, bbox_transform_inv, clip_boxes

# from faster_rcnn.model.configs import config

RPN_PRE_NMS_TOP_N = 12000
RPN_POST_NMS_TOP_N = 2000
RPN_NMS_THRESH = 0.7
RPN_MIN_SIZE = 8
 
class _ProposalLayer(nn.Module):
    """
    Get object detection proposals by applying estimated bounding=box 
    transformations to a set of regular boxes (called "anchors")
    """
    def __init__(self, feat_stride, scales, ratios):
        super(_ProposalLayer, self).__init__()
        self._feat_stride = feat_stride
        self.device = torch.device("cuda")
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales), 
                                                ratios=np.array(ratios)))
        self._num_anchors = self._anchors.size(0)

    def forward(self, rpn_cls_probs, rpn_pred_bboxes, im_shapes, cfg_key):
        """
        Input:
        rpn_cls_probs: torch Tensor: shape batch_size, nc_scores, 
        rpn_pred_bboxes: torch Tensor: shape batch_size, nc_
        Algorithm:
        for each (H, W) location i
            generate a anchor boxes centered on cell i
            apply predictied bboxes deltas at cell i to each of this A anchors

        """

        # rpn_cls_probs shape is [batch_size, nc_score_out, 7, 7]
        scores = rpn_cls_probs[:, self._num_anchors:, :, :]
        pre_nms_topN  = RPN_PRE_NMS_TOP_N
        post_nms_topN = RPN_POST_NMS_TOP_N
        nms_thresh    = RPN_NMS_THRESH
        min_size      = RPN_MIN_SIZE
        
        bboxes_deltas = rpn_pred_bboxes

        batch_size = rpn_pred_bboxes.size(0)
        feat_height, feat_width = rpn_pred_bboxes.size(2), rpn_pred_bboxes.size(3)
        shift_x = np.arange(0, feat_width)*self._feat_stride # 7x1
        shift_y = np.arange(0, feat_height)*self._feat_stride # 7x1
        shift_x, shift_y = np.meshgrid(shift_x, shift_y) # 7x7
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        
        shifts = shifts.contiguous().type_as(scores).float() # 49x4

        A = self._num_anchors
        K = shifts.size(0)

        self._anchors = self._anchors.type_as(scores) # 9x4 tensor
        anchors =  self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)

        # Transpose and reshape predicted box transformation to get them
        # into the same order as of anchors boxes
        bboxes_deltas = rearrange(bboxes_deltas, "b d h w -> b h w d").contiguous()
        bboxes_deltas = bboxes_deltas.view(batch_size, -1, 4)

        # Do the same for scors
        scores = rearrange(scores, "b d h w -> b h w d").contiguous()
        scores = rearrange(scores, "b d h w -> b (h w d)")

        # Convrt anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bboxes_deltas)
        proposals = clip_boxes(proposals, im_shapes, batch_size)

        return