from torch import nn
from model.cbrelu import CBRelu

class Skips(nn.Module):
    def __init__(self, cfg, input, id):
        super().__init__()
        # Common configs
        self.cfg = cfg
        self.match_fc = cfg['__fc_features__']
        batch, channels, width, height = input.shape

        # self.skips = CBRelu(channels, self.match_fc, 1, 1, 0)
      
        self.skips = nn.Sequential(
            CBRelu(channels, self.match_fc//8, 1, 1, 0),
            CBRelu(self.match_fc//8, self.match_fc, 1, 1, 0),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

        # self.attention = nn.Sequential(
        #     CBRelu(self.match_fc, self.match_fc, 1, 1, 0),
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Sigmoid()
        # )

    def forward(self, input):
        x = self.skips(input)

        # x = self.attention(x) * x
        return x