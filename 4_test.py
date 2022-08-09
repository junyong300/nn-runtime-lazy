
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
        self.model = LazyNet(cfg).to(self.device)
        filename = '{}_{}_{}.pth'.format(
            cfg['dataset'], cfg['backbone'], cfg['mode'])
        self.model.load_state_dict(
            torch.load(cfg['save-dir']+filename), strict=False)
    
    def test(self):
        self.model.eval()
        self.model.train_mode=False
        pbar = tqdm(self.test_loader)
        epochs = 2000

        # Start validation
        total = 0
        correct = 0

        start_time = time.time()

        for i, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.type(torch.LongTensor).to(self.device)
            total += images.shape[0]

            with torch.no_grad():
                logits = self.model(images)
                conf, pred = torch.max(logits, dim=1)

            correct += pred.eq(labels).sum().item()
            pbar.set_description(
                "[{}]:accc:{:.2f}%".format(
                    self.cfg['mode'], correct*100/total))
            if i == 2000:
                break

        latency = time.time() - start_time
        print("total time elasped: ",latency / epochs," ms" )
        print("skips count:", self.model.exit_list)

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
    trainer.test()



if __name__ == '__main__':
    main()