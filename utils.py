import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

def convert_cellboxes_2_imgboxes(preds, S=7):
    """
    Convert
    """
    preds = preds.to("cpu")
    batch_size = preds.shape[0]
    import pdb; pdb.set_trace()
    preds = preds.reshape(batch_size, S, S, 30)
    boxes_1 = preds[..., 21:25]
    boxes_2 = preds[..., 26:30]
    scores = torch.cat(
        (preds[..., 20].unsqueeze(0), preds[..., 25].unsqueeze(0)), dim=0
    )

    scores = torch.cat((preds[..., 20].unsqueeze(0), preds[..., 25].unsqueeze(0)), dim=0)
    best_box_indices = scores.argmax(0).unsqueeze(-1)
    best_boxes = boxes_1 * (1 - best_box_indices) + boxes_2 * best_box_indices
    
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = preds[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(preds[..., 20], preds[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds

def cellboxes_to_boxes(outputs, S=7):
    converted_pred = convert_cellboxes_2_imgboxes(outputs).reshape(outputs.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(outputs.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def intersection_over_union(box_preds, box_labels, box_mode=None):
    assert box_mode is not None, "box mode must be cxcywh or xyxy"
    if box_mode == "cxcywh":
        pred_x1 = box_preds[..., 0: 1] - box_preds[..., 2: 3]/2
        pred_y1 = box_preds[..., 1: 2] - box_preds[..., 3: 4]/2
        pred_x2 = box_preds[..., 0: 1] + box_preds[..., 2: 3]/2
        pred_y2 = box_preds[..., 1: 2] + box_preds[..., 3: 4]/2
        tgt_x1 = box_labels[..., 0: 1] - box_labels[..., 2: 3]/2
        tgt_y1 = box_labels[..., 1: 2] - box_labels[..., 3: 4]/2
        tgt_x2 = box_labels[..., 0: 1] + box_labels[..., 2: 3]/2
        tgt_y2 = box_labels[..., 1: 2] + box_labels[..., 3: 4]/2
        x1 = torch.max(pred_x1, tgt_x1)
        y1 = torch.max(pred_y1, tgt_y1)
        x2 = torch.max(pred_x2, tgt_x2)
        y2 = torch.max(pred_y2, tgt_y2)
        # .clamp(0) is for the case when they do not intersect
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        pred_area = abs((pred_x2 - pred_x1) * (pred_y2 - pred_y1))
        tgt_area = abs((tgt_x2 - tgt_x1) * (tgt_y2 - tgt_y1))
        return intersection / (pred_area + tgt_area - intersection + 1e-6)
    else:
        raise NotImplementedError
    return 

def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision 
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """
    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)