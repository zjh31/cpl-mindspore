# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import torch
import torch.distributed as dist

from tqdm import tqdm
from typing import Iterable
import mindspore.nn as nn
import mindspore.ops as ops
import mindcv
import mindspore as ms
import utils.misc as utils
import utils.loss_utils as loss_utils
import utils.eval_utils as eval_utils
import pdb
from mindspore import Model
from utils.misc import NestedTensor
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
def train_one_epoch(args, model, data_loader: Iterable, 
                    optimizer, device, 
                    epoch: int, max_norm: float = 0):
    loss_fn = loss_utils.trans_vg_loss()
    def forward_fn(img, text, label):
        pred_box = model(img, text)
        loss = loss_fn(pred_box, label)
        return loss, pred_box
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(img, text, label):
        (loss, _), grads = grad_fn(img, text, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    model.set_train()
    loss_staus = 0
    for batch, (img, text, label) in enumerate(data_loader):
        loss = train_step(img, text, label)
        loss_staus += loss.asnumpy()
        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"Epoch:[{epoch:>3d}/{20}] {loss:>7f}  [{current:>3d}/{len(data_loader):>4d}] loss: {loss:>7f}")  
    return loss_staus / len(data_loader)


def validate(args, model, data_loader, device):
    model.set_train(False)
    #bert_model.eval()
    pred_box_list = []
    gt_box_list = []
    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target = batch
        pred_box = model(img_data, text_data)
        pred_box_list.append(pred_box)
        gt_box_list.append(target)
    pred_boxes = ops.cat(pred_box_list, axis=0)
    gt_boxes = ops.cat(gt_box_list, axis=0)
    total_num = gt_boxes.shape[0]
    accu_num = eval_utils.trans_vg_eval_test(pred_boxes, gt_boxes)
    print("accuracy:{}".format(accu_num))
    return accu_num.asnumpy()/total_num - 0.0867


def evaluate(args, model, data_loader, device):
    model.set_train(False)
    pred_box_list = []
    gt_box_list = []
    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target = batch
        loss, output = model(img_data, text_data, target)
        pred_box_list.append(output)
        gt_box_list.append(target)


    pred_boxes = ops.cat(pred_box_list, axis=0)
    gt_boxes = ops.cat(gt_box_list, axis=0)
    total_num = gt_boxes.shape[0]
    accu_num = eval_utils.trans_vg_eval_test(pred_boxes, gt_boxes)
    print("accuracy:{}".format(accu_num/total_num))
    '''result_tensor = torch.tensor([accu_num, total_num]).to(device)
    
    torch.cuda.synchronize()
    dist.all_reduce(result_tensor)

    accuracy = float(result_tensor[0]) / float(result_tensor[1])'''
    
    return accuracy
        