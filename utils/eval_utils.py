import torch
import numpy as np

from utils.box_utils import bbox_iou, xywh2xyxy, ms_xywh2xyxy, ms_xyxy2xywh
import mindspore.nn as nn
import mindspore.ops as ops
import mindcv
import mindspore as ms
import pdb
def trans_vg_eval_val(pred_boxes, gt_boxes):
    batch_size = pred_boxes.shape[0]
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu = torch.sum(iou >= 0.5) / float(batch_size)

    return iou, accu

def trans_vg_eval_test(pred_boxes, gt_boxes):
    pred_boxes = ms_xywh2xyxy(pred_boxes)
    pred_boxes = ops.clamp(pred_boxes, 0, 1)
    gt_boxes = ms_xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu_num = ops.sum(iou >= 0.5)

    return accu_num