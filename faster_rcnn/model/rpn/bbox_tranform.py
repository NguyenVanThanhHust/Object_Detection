import torch
import numpy as np

# NOTE: follow below
# https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/boxes.py
# not original from faster-rcnn pytorch

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh),1)

    return targets
    
def bbox_transform_inv(boxes, bbox_deltas):
    """
    
    """
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0

    ctr_x = boxes[:, :, 0] + 0.5*widths
    ctr_y = boxes[:, :, 1] + 0.5*heights

    dx = bbox_deltas[:, :, 0::4]
    dy = bbox_deltas[:, :, 1::4]
    dw = bbox_deltas[:, :, 2::4]
    dh = bbox_deltas[:, :, 3::4]
    try:
        pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
        pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    except Exception as e:
        import pdb; pdb.set_trace()
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)

    pred_boxes = bbox_deltas.clone()
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h
    return pred_boxes
    
def clip_boxes(boxes, im_shape, batch_size):
    for i in range(batch_size):
        boxes[i, :, 0::4].clamp_(0, im_shape[i, 1] - 1) 
        boxes[i, :, 1::4].clamp_(0, im_shape[i, 0] - 1) 
        boxes[i, :, 2::4].clamp_(0, im_shape[i, 1] - 1) 
        boxes[i, :, 3::4].clamp_(0, im_shape[i, 0] - 1) 
    return boxes