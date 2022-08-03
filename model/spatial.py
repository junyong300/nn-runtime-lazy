from torch import nn
from model.cbrelu import CBRelu

class Spatial(nn.Module):
    """Spatial Analysis & Decision Making for exiting"""
    def __init__(self, cfg):
        super().__init__()
        img_size        = cfg['img_size']
        in_channels     = 3
        inter_channels  = cfg['skips']['inter_ch']
        inter_features  = cfg['__fc_features__']
        num_class       = cfg['num_class']
        out_size        = img_size / cfg['compress_ratio']
        
        # self.transform = nn.Sequential(
        #     CBRelu(in_channels, inter_channels, 7, 2, 3),
        #     CBRelu(inter_channels, inter_channels, 3, 2, 1),
        #     CBRelu(inter_channels, inter_channels, 1, 2, 0))

        self.transform = nn.Sequential(
            CBRelu(in_channels, inter_channels, 7, 4, 3),
            CBRelu(inter_channels, inter_channels, 3, 4, 0),
            CBRelu(inter_channels, inter_channels, 3, 2, 1))

        self.attention = nn.Sequential(
            CBRelu(inter_channels, inter_channels, 1, 1, 0),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

        self.channels = nn.Sequential(
            CBRelu(inter_channels, inter_features, 1, 1, 0)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),            
            nn.Linear(inter_features, num_class))

    def forward(self, input):
        """Forward"""
        repr = self.transform(input)
        attention = self.attention(repr)
        repr += repr * attention
        repr = self.channels(repr)
        logits = self.classifier(repr)
        return repr, logits