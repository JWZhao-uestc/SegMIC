import os
import torch


"""
GPU wrappers
"""

def set_gpu_mode(mode, gid):
    gpu_id = gid
    use_gpu = mode
    device = torch.device(f"cuda:{gpu_id}" if use_gpu else "cpu")
    torch.backends.cudnn.benchmark = True
    return device