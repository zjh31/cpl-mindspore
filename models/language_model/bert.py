# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import torch
from torch import nn
from utils.misc import NestedTensor
#from pytorch_pretrained_bert.modeling import BertModel
from mindformers import BertModel, AutoConfig
import mindspore.nn as nn
import mindspore.ops as ops
import mindcv
import mindspore as ms
import pdb
class BERT(nn.Cell):
    def __init__(self, name: str, train_bert: bool, hidden_dim: int, max_len: int, enc_num):
        super().__init__()
        if name == 'bert_base_uncased':
            self.num_channels = 768
        else:
            self.num_channels = 1024
        self.enc_num = enc_num
        bert_config = AutoConfig.from_pretrained("bert_base_uncased")
        bert_config.seq_length = 20
        self.bert = BertModel(bert_config)

        if not train_bert:
            for parameter in self.bert.get_parameters():
                parameter.requires_grad = False

    def construct(self, tensor_list: NestedTensor):
        if self.enc_num > 0:
            all_encoder_layers, _, _ = self.bert(tensor_list.tensors, token_type_ids=None, input_mask=tensor_list.mask)
            # use the output of the X-th transformer encoder layers
            xs = all_encoder_layers
        else:
            xs = self.bert.embeddings.word_embeddings(tensor_list.tensors)
        
        mask = tensor_list.mask.bool()
        mask = ~mask
        out = NestedTensor(xs, mask)

        return out


def build_bert(args):
    train_bert = args.lr_bert > 0
    bert = BERT(args.bert_model, train_bert, args.hidden_dim, args.max_query_len, args.bert_enc_num)
    return bert
