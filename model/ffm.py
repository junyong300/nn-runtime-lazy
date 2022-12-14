import torch
from torch import nn
import torch.nn.functional as F
from model.cbrelu import CBRelu
class FeatureFusion(nn.Module):
    def __init__(self, shape, reduction=1):
        """feature fusion module"""
        super().__init__()
        batch, channel, width, height = shape
        self.width = width
        self.height= height
        # self.conv1 = CBRelu(channel*2, channel, 1, 1, 0)

        self.conv1 = nn.Sequential(
            CBRelu(channel*2, channel//8, 1, 1, 0),
            CBRelu(channel//8, channel, 1, 1, 0),
            )


        # self.resize = nn.AdaptiveAvgPool2d((width, height))
        # self.resize = F.interpolate(size=(width, height))
        
        # self.atten = nn.Sequential(
        #     CBRelu(channel, channel//reduction, 1, 1, 0),
        #     CBRelu(channel//reduction, channel, 1, 1, 0),
        #     nn.Sigmoid())

        self.resize = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x1, x2):
        """forward"""
        # x1 = self.resize(x1)
        # x2 = self.resize(x2)
        # print("[ffm]x1", x1.shape)
        # print("[ffm]x1", x2.shape)

        # x1 = F.interpolate(
        #     input=x1, size=(self.width, self.height), mode='nearest')
        
        # x2 = F.interpolate(
        #     input=x2, size=(self.width, self.height), mode='nearest')
        
        x1 = self.resize(x1)
        x2 = self.resize(x2)
        

        x = torch.cat([x1, x2], dim=1)
        # print("[ffm]x",x.shape)
        mid = self.conv1(x)
        # print("[ffm]mid",mid.shape)
        # att = self.atten(mid)
        # print("[ffm]att",att.shape)

        # out = mid + mid * att
        # return out
        return mid