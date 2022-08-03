"""My version of Lazy Network"""
import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from model.backbone import get_backbone, get_plan, get_modulelist
from model.ffm import FeatureFusion
from model.cbrelu import CBRelu
from model.skips import Skips
from model.spatial import Spatial
import copy

class LazyNet(nn.Module):
    """load backbone into model"""
    def __init__(self, cfg):
        """initialization function for lazy"""
        super().__init__()
        # Init configuration file
        self.cfg = cfg
        
        # Data config
        self.num_class  = cfg['num_class']
        self.img_size   = cfg['img_size']
        self.device     = cfg['device']

        # Backbone Config
        self.bb_name    = cfg['backbone']
        self.lazy_num   = cfg['lazy_num']
        backbone        = get_backbone(cfg)
        plan            = get_plan(backbone, cfg['backbone'])
        
        head_list       = get_modulelist(
            plan['head'], start=plan['head_start'], end=plan['head_end'])

        mid_list       = get_modulelist(
            plan['mid'], start=plan['mid_start'], end=plan['mid_end'])

        tail_list       = get_modulelist(
            plan['tail'], start=plan['tail_start'], end=plan['tail_end'])

        self.construct(backbone, head_list, mid_list, tail_list)

        # Set Spatial Analysis
        self.spatial    = Spatial(cfg)

        self.train      = True
        
    def construct(self, backbone, head_list, mid_list, tail_list):
        """construct lazy parts using plan"""
        id = 0
        N = self.lazy_num
        input = torch.randn(1, 3, self.cfg['img_size'], self.cfg['img_size'])
        # Create Head layers
        self.head_layer = nn.Sequential(*head_list)
        input = self.head_layer(input)

        # Create Mid Layers
        print('---------------------------------------------------')
        self.feats_layers = nn.ModuleList([])
        self.skip_layers  = nn.ModuleList([])
        print("split feats={}, using N={} Lazy Entries"
              .format(len(mid_list), N))
        if N == 1:
            N += 1            
        div = len(mid_list) / N
        div = int(div)
        print("divide size:", div)
        split_list = lambda test_list, \
            x: [test_list[i:i+x] for i in range(0, len(test_list), x)]
        final_list = split_list(mid_list, div)
        print("Constructing head-body-tail layers with lazy entries")
        print("<head layer>")
        print('     || ')

        for x in range(N):              
            for y in range(len(final_list[x])):
                if y < len(final_list[x])-1:
                    print('[feat layer]')
                else:
                    print('[feat layer] -> [lazy #{}]'.format(x))
            print('     || ')
            self.feats_layers.append(nn.Sequential(*final_list[x]))
            input = self.feats_layers[x].forward(input)
            self.skip_layers.append(Skips(cfg=self.cfg, input=input, id=id))
            id += 1

        # Create Fetc layers
        self.fetc_layers = nn.ModuleList([])
        x += 1
        for y in range(x, len(final_list)):
            self.fetc_layers.append(nn.Sequential(*final_list[x]))
            print('[fetc layer]')
        
        for fetc in self.fetc_layers:
            input = fetc(input)
        
        print('     || ')
        print("<tail layer>")
        print('---------------------------------------------------')
        print("Model Set Complete!")
        
        self.ff_layer = FeatureFusion(input.shape, reduction=2)

        self.tail_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(
                self.cfg['__fc_features__'], self.cfg['num_class'], bias=True)
        )
        # input = self.tail_layers(input)
        # return input

    def forward(self, input):
        # Spatial Analysis
        repr, logits = self.spatial(input)
        
        # Head Inference
        x = self.head_layer(input)

        # Mid Inference
        ss = []
        for i, (feat, skip) in enumerate(
            zip(self.feats_layers, self.skip_layers)):
            x = feat(x)
            s = skip(x)
            ss.append(s)
        
        # Fetc Inference
        for fetc in self.fetc_layers:
            x = fetc(x)

        # Feature Fusion
        vals = []
        for s_iter in ss:
            val = self.ff_layer(s_iter, repr)
            val = self.tail_layers(val)
            vals.append(val)
        return repr, logits, vals

    # def forward(self, input):
    #     print("[init] input:\t",input.shape)
    #     repr, logits = self.spatial(input)
    #     print("[spac] repr:\t",repr.shape)

    #     x = self.head_layer(input)
    #     print('[head]x.shape:\t',x.shape)

    #     ss = []

    #     for i, (feat, skip) in enumerate(
    #         zip(self.feats_layers, self.skip_layers)):
    #         print('[feat#'+str(i)+']pre-x:\t',x.shape)
    #         x = feat(x)
    #         print('[feat#'+str(i)+']post-x:\t',x.shape)
    #         s = skip(x)
    #         ss.append(s)
    #         print('[skip#'+str(i)+']post-x:\t',s.shape)
        
    #     for fetc in self.fetc_layers:
    #         print('[fetc]pre-x:\t',x.shape)
    #         x = fetc(x)
    #         print('[fetc]post-x:\t',x.shape)

    #     vals = []

    #     for s_iter in ss:
    #         val = self.ff_layer(s_iter, x)
    #         val = self.tail_layers(val)
    #         vals.append(val)
    #     return vals
    #     # val = self.ff_layer(s, x)
    #     # print('[ffm]val.shape:\t',val.shape)

    #     # val = self.tail_layers(x)
    #     # print('[tail]val.shape:\t',val.shape)

    #     # return val  