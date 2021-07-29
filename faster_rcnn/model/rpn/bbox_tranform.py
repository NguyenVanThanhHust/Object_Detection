import torch
import numpy as np

# NOTE: follow below
# https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/boxes.py
# not original from faster-rcnn pytorch

def bbox_transform(boxes, deltas):
    """
    Forward transform that maps proposals to predited ground truth
    boxes 
    """
    if boxes.shape[0] == 0:
        return np.zeros(0, deltas.shape[0], dtype=deltas.shape)
    return 
    
def bbox_transform_inv(anchors, bbox_deltas, batch_size):
    """
    
    """
    return 
    
def clip_boxes(boxes, im_shape, batch_size):
    for i in range(batch_size):
        boxes[] = 

    return boxes