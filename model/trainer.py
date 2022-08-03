"""Trainer Class"""
import torch
from torch import nn
import torch.nn.functional as F
from data import get_dataset
from torch.optim import Adam

class Trainer():
    """Trainer Class containing train, validation, calibration"""
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg['device'])
        loaders = get_dataset(cfg)
        self.trainloader = loaders[0]
        self.valloader = loaders[1]
        self.testloader = loaders[2]

        self.criterion = nn.CrossEntropyLoss()
        
        
        