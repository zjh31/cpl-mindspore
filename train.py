import argparse
import datetime
import json
import random
import time
import math

import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler
import pdb
import datasets
import utils.misc as utils
from models import build_model
from datasets import build_dataset
from engine import train_one_epoch, validate
import mindspore.nn as nn
import mindspore.ops as ops
import mindcv
import mindspore as ms
from mindspore.experimental import optim
import warnings
warnings.filterwarnings("ignore")

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_bert', default=1e-5, type=float)
    parser.add_argument('--lr_visu_cnn', default=1e-5, type=float)
    parser.add_argument('--lr_visu_tra', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr_power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--clip_max_norm', default=0., type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr_scheduler', default='cosine', type=str)
    parser.add_argument('--lr_drop', default=60, type=int)
    
    # Augmentation options
    parser.add_argument('--aug_blur', action='store_true',
                        help="If true, use gaussian blur augmentation")
    parser.add_argument('--aug_crop', action='store_true',
                        help="If true, use random crop augmentation")
    parser.add_argument('--aug_scale', action='store_true',
                        help="If true, use multi-scale augmentation")
    parser.add_argument('--aug_translate', action='store_true',
                        help="If true, use random translate augmentation")

    # Model parameters
    parser.add_argument('--model_name', type=str, default='TransVG',
                        help="Name of model to be exploited.")
    
    # DETR parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    parser.add_argument('--imsize', default=640, type=int, help='image size')
    parser.add_argument('--emb_size', default=512, type=int,
                        help='fusion module embedding dimensions')

    # Transformers in two branches
    parser.add_argument('--bert_enc_num', default=12, type=int)
    parser.add_argument('--detr_enc_num', default=6, type=int)

    # Vision-Language Transformer
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_hidden_dim', default=256, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=6, type=int,
                        help='Number of encoders in the vision-language transformer')

    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='./ln_data/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='/home/zhangjiahua/Code/Pseudo-Q/data/data',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='unc', type=str,
                        help='referit/unc/unc+/gref/gref_umd')
    parser.add_argument('--max_query_len', default=20, type=int,
                        help='maximum time steps (lang length) per batch')
    
    # dataset parameters
    parser.add_argument('--output_dir', default='./outputs/test',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--detr_model', default='/home/zhangjiahua/Code/reproduce/unc_detr_1.ckpt', type=str, help='detr model')
    parser.add_argument('--bert_model', default='bert_base_uncased', type=str, help='bert model')
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # build model
    ms.set_context(device_target='GPU')
    model = build_model(args)
    #model.to(device)

    model_without_ddp = model
    #pdb.set_trace()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.get_parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    visu_cnn_param = [p for n, p in model_without_ddp.parameters_and_names() if (("visumodel" in n) and ("backbone" in n) and p.requires_grad)]
    visu_tra_param = [p for n, p in model_without_ddp.parameters_and_names() if (("visumodel" in n) and ("backbone" not in n) and p.requires_grad)]
    text_tra_param = [p for n, p in model_without_ddp.parameters_and_names() if (("textmodel" in n) and p.requires_grad)]
    rest_param = [p for n, p in model_without_ddp.parameters_and_names() if (("visumodel" not in n) and ("textmodel" not in n) and p.requires_grad)]

    rest_lr = nn.CosineDecayLR(min_lr=1e-6, max_lr=args.lr, decay_steps=4000)
    backbone_lr = nn.CosineDecayLR(min_lr=1e-6, max_lr=args.lr_bert, decay_steps=4000)


    param_list = [
        {"params": rest_param, "lr": rest_lr},
                  {"params": visu_tra_param, "lr": backbone_lr},
                   {"params": text_tra_param, "lr": backbone_lr},
                   {"params": visu_cnn_param, "lr": backbone_lr}

                   ]
    # using RMSProp or AdamW
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = nn.AdamWeightDecay(param_list, learning_rate=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise ValueError('Lr scheduler type not supportted ')

    # build dataset
    dataset_train = build_dataset('train', args)
    dataset_val   = build_dataset('val', args)
    
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        sampler_val   = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val   = torch.utils.data.SequentialSampler(dataset_val)
    
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)


    if args.resume:
        param_dict = ms.load_checkpoint(args.resume)
        missing_keys, unexpected_keys = ms.load_param_into_net(model, param_dict)
        print('Missing keys when loading model:')
        print(missing_keys)
    elif args.detr_model is not None:
        param_dict = ms.load_checkpoint(args.detr_model)
        missing_keys, unexpected_keys = ms.load_param_into_net(model.visumodel, param_dict)
        print('Missing keys when loading detr model:')
        print(missing_keys)
        param_dict = ms.load_checkpoint('bert.ckpt')
        missing_keys, unexpected_keys = ms.load_param_into_net(model.textmodel, param_dict)
        print('Missing keys when loading bert model:')
        print(missing_keys)
    
    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(str(args) + "\n")

    print("Start training")
    start_time = time.time()
    best_accu = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            args, model, data_loader_train, optimizer, device, epoch, args.clip_max_norm
        )
        #lr_scheduler.step()

        val_stats = validate(args, model, data_loader_val, device)
        
        log_stats = {
                     'epoch': epoch,
                     'n_parameters': n_parameters,
                     'loss': train_stats,
                     'acc': val_stats
                     }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.ckpt']
            # extra checkpoint before LR drop and every 10 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.ckpt')
            if val_stats['accu'] > best_accu:
                checkpoint_paths.append(output_dir / 'best_checkpoint.ckpt')
                best_accu = val_stats['accu']
            
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'val_accu': val_stats['accu']
                }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TransVG training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
