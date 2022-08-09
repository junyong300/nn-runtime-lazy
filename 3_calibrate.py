
import os
from argparse import ArgumentParser
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils import read_yaml, save_checkpoint, ConfidenceHistogram, ReliabilityDiagram
from model.lazy import LazyNet
from data import get_dataset
from tqdm import tqdm
from torch import optim

def parse_args():
    """Parse Argument"""
    parser = ArgumentParser(description='Edge Lazy Inference Prototype')
    parser.add_argument('-c', type=str, default="./cfgs/c10/1-e0.yml")
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

        # Load Model
        self.model = LazyNet(cfg).to(self.device)
        filename = '{}_{}_{}.pth'.format(
            cfg['dataset'], cfg['backbone'], cfg['mode'])
        self.model.load_state_dict(
            torch.load(cfg['save-dir']+filename), strict=False)

        # Training Optimizer
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def calibrate(self):
        self.model.eval()
        # Start Training
        pbar = tqdm(self.val_loader)
        logitss = torch.zeros((0))
        scaleds = torch.zeros((0))
        labelss = torch.zeros(0).long()
        self.model.spatial.temperature = nn.Parameter(
            torch.Tensor([1.0]).to(self.device)
        )
        for i, (images, labels) in enumerate(pbar):

            skip_losses = torch.zeros(self.model.lazy_num+1).to(self.device)
            images = images.to(self.device)
            labels = labels.type(torch.LongTensor)

            with torch.no_grad():
                vals, logits = self.model(images)

                labelss = torch.cat((labelss, labels), dim=0)
                logitss = torch.cat((logitss, logits.cpu().detach()), dim=0)

        temperature = nn.Parameter(torch.Tensor([1]))
        optimizer = optim.LBFGS(
            [temperature], lr=0.01, 
            max_iter=10000, line_search_fn='strong_wolfe')

        def eval():
            optimizer.zero_grad()
            loss = self.criterion(
                torch.div(logitss, temperature), labelss)
            loss.backward()
            return loss
        optimizer.step(eval)
        scaled = torch.div(logitss, temperature)       

        labels_np = labelss.cpu().detach().numpy()
        scaled_np = scaled.cpu().detach().numpy()
        logits_np = logitss.cpu().detach().numpy()

        conf_hist = ConfidenceHistogram()
        plt = conf_hist.plot(
            logits_np, labels_np, title="Conf. Logits")
        name = 'checkpoints/conf_histogram_before_.png'
        plt.savefig(name, bbox_inches='tight')

        plt = conf_hist.plot(
            scaled_np, labels_np, title="Confidence Histogram")
        name = 'checkpoints/conf_histogram_after_.png'
        plt.savefig(name, bbox_inches='tight')

        rel_diagram = ReliabilityDiagram()
        plt = rel_diagram.plot(
            logits_np, labels_np, title="Reliability Logits")
        name = 'checkpoints/reliability_before.png'
        plt.savefig(name, bbox_inches='tight')

        rel_diagram = ReliabilityDiagram()
        plt = rel_diagram.plot(
            scaled_np, labels_np, title="Reliability Graph")
        name = 'checkpoints/reliability_after.png'
        plt.savefig(name, bbox_inches='tight')

        self.model.temperature = temperature
        print(self.model.temperature)
        save_checkpoint(self.model, self.cfg, 0)


def main():
    """Main function"""
    args = parse_args()
    cfg = read_yaml(args.c)
    trainer = Trainer(cfg)
    trainer.calibrate()


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()

###https://madebyollin.github.io/convnet-calculator/