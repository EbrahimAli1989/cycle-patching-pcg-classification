# --------------------------------------------------------
# Swin Transformer / FocalNet
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
#
# Small stateless helpers, unchanged from custom_utils.get_grad_norm /
# reduce_tensor. `reduce_tensor` is unused by `Trainer` (distributed training
# is not wired up in this codebase -- see train.py's commented-out
# `torch.distributed` calls in the original scripts) but is kept for parity.

import torch
import torch.distributed as dist


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt
