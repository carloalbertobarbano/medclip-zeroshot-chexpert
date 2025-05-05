"""
Author: Carlo Alberto Barbano (carlo.barbano@unito.it)
Date: 23/04/24
"""
import random
import os
import numpy as np
import torch

from contextlib import suppress
from functools import partial


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


def partial_forward(model: VisionTransformer, start_block, x):
    x = model.transformer.resblocks[start_block:](x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = model.ln_post(x[:, 0, :])
    if model.proj is not None:
        x = x @ model.proj
    return x


def print_trainable_parameters(model, label):
    parameters, trainable = 0, 0

    for _, p in model.named_parameters():
        parameters += p.numel()
        trainable += p.numel() if p.requires_grad else 0

    print(f"{label} trainable parameters: {trainable:,}/{parameters:,} ({100 * trainable / parameters:.2f}%)")


def swap_layer_class(model, source_class, target_class, verbose=False):
    for name, module in model.named_modules():
        if isinstance(module, source_class):
            if verbose:
                print("Converting", name, module, "to", target_class)
            module.__class__ = target_class


def get_autocast(precision, device_type='cuda'):
    if precision =='amp':
        amp_dtype = torch.float16
    elif precision == 'amp_bfloat16' or precision == 'amp_bf16':
        amp_dtype = torch.bfloat16
    else:
        return suppress

    return partial(torch.amp.autocast, device_type=device_type, dtype=amp_dtype)


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype