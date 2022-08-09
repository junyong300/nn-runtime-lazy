
"""LazyNet Model Converter \
    from Pytorch to ONNX, TFLite, and Pytorch Mobile
"""
import torch
import torchvision
import model
from utils import *
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as  np
import yaml
from model.lazy import LazyNet
from argparse import ArgumentParser
from data import get_dataset
from tqdm import tqdm
import time
import utils
from torch.profiler import profile, record_function, ProfilerActivity
from model.backbone import get_backbone

def parse_args():
    """Argument Parser function"""
    parser = ArgumentParser(
        description='convertion to onnx, tf, tflite')
    parser.add_argument('-c', type=str, default="./cfgs/c10/1-e0.yml",
                        help='model name (default: efficientnet-b0)')
    return parser.parse_args()

class Trainer():
    """Trainer Class containing train, validation, calibration"""
    def __init__(self, cfg):
        # Init config
        self.cfg = cfg
        self.device = torch.device(cfg['device'])
        
        # Get Dataset
        loaders = get_dataset(cfg)
        self.train_loader = loaders[0]
        self.val_loader = loaders[1]
        self.test_loader = loaders[2]

        # Set Model
        self.model = get_backbone(cfg).to(self.device)
        # self.model = LazyNet(cfg).cpu()
        # filename = '{}_{}_{}.pth'.format(
        #     cfg['dataset'], cfg['backbone'], cfg['mode'])
        # self.model.load_state_dict(
        #     torch.load(cfg['save-dir']+filename), strict=False)
    
    def profile(self):
        self.model.eval()
        self.model.train_mode=False
        pbar = tqdm(self.test_loader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(self.device)

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True, profile_memory=True) as prof:
                with record_function("model_inference"):
                    self.model(images)
            if i == 1:
                break
        print(
            prof.key_averages().table(
                sort_by="cpu_memory_usage"))

def main():
    """convert pytorch model to:
    pytorch mobile
    pytorch mobile optimized for mobile
    onnx
    tensorflow
    tflite
    """
    args = parse_args()
    cfg = read_yaml(args.c)
    trainer = Trainer(cfg)
    trainer.profile()



if __name__ == '__main__':
    main()