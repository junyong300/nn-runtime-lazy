import torch
from model.backbone import get_backbone
from model.lazy import LazyNet


def get_size(module):
    param_size = 0
    for param in module.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in module.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    print('module size: {:.3f}MB'.format(size_all_mb))

    