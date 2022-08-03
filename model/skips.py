from torch import nn
from model.cbrelu import CBRelu

class Skips(nn.Module):
    def __init__(self, cfg, input, id):
        super().__init__()
        # Common configs
        self.cfg = cfg
        self.match_fc = cfg['__fc_features__']
        batch, channels, width, height = input.shape
       
        self.skips = CBRelu(channels, self.match_fc, 1, 1, 0)

    def forward(self, input):
        x = self.skips(input)
        return x