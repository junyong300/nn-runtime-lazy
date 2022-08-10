import torch
from model.backbone import get_backbone
from model.lazy import LazyNet
from utils import read_yaml

def get_size(module):
    param_size = 0
    for param in module.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in module.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    print('module size: {:.3f}MB'.format(size_all_mb))



def get_num(module):
    param_num = 0
    for param in module.parameters():
        param_num += param.nelement()
    print("param_num: {:,}".format(param_num))
    
cfg = read_yaml("cfgs/c10/3.yml")
backbone = get_backbone(cfg)
from size import get_size, get_num
lazy=LazyNet(cfg)
get_num(backbone)
get_num(lazy.ff_layer)
get_num(lazy.skip_layers)
get_num(lazy)
