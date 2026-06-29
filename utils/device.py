import os

import torch


def resolve_device(requested=None):
    device = requested or os.environ.get('CONDITAR_DEVICE', 'auto')
    if device == 'auto':
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and not torch.cuda.is_available():
        return 'cpu'
    return device
