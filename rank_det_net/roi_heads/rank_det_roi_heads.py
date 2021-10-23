from fvcore.common.registry import Registry
from typing import Dict, List, Tuple, Union
import torch
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.layers import ShapeSpec

from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.layers import (
    ShapeSpec, 
)


@ROI_HEADS_REGISTRY.register()
class RankROIHeads(StandardROIHeads): 
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(RankROIHeads, self).__init__(cfg, input_shape)
        self.config=cfg


    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}



class RankDetNetOutputLayers(FastRCNNOutputLayers):
    def __init__(self, input_shape: ShapeSpec, *, box2box_transform, num_classes: int, test_score_thresh: float = 0, test_nms_thresh: float = 0.5, test_topk_per_image: int = 100, cls_agnostic_bbox_reg: bool = False, smooth_l1_beta: float = 0, box_reg_loss_type: str = "smooth_l1", loss_weight: Union[float, Dict[str, float]] = 1):
        super().__init__(input_shape, box2box_transform, num_classes, test_score_thresh=test_score_thresh, test_nms_thresh=test_nms_thresh, test_topk_per_image=test_topk_per_image, cls_agnostic_bbox_reg=cls_agnostic_bbox_reg, smooth_l1_beta=smooth_l1_beta, box_reg_loss_type=box_reg_loss_type, loss_weight=loss_weight)
