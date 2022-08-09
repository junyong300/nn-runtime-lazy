from argparse import ArgumentParser
import torch
from torch import nn
from torch.nn import functional as F
from utils import read_yaml
from model.lazy import LazyNet

def parse_args():
    """Parse Argument"""
    parser = ArgumentParser(description='Edge Lazy Inference Prototype')
    parser.add_argument('--configs', type=str, default="./configs/base.yml")
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    cfg = read_yaml(args.configs)
    model = LazyNet(cfg)
    
    x = torch.rand((1,3,cfg['img_size'],cfg['img_size']))
    result = model(x)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()

###https://madebyollin.github.io/convnet-calculator/