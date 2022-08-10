from argparse import ArgumentParser
import torch
from torch import nn
from torch import fx
from model.backbone import get_backbone, get_plan, get_modulelist
from utils import read_yaml
from model.cbrelu import CBRelu
from torch.nn import functional as F
def parse_args():
    """Parse Argument"""
    parser = ArgumentParser(description='Edge Tomato Pest and \
        Disease Semantic Segmentation Training')
    parser.add_argument('--configs', type=str, default="./configs/base.yml")
    return parser.parse_args()

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
        
        self.transform = nn.Sequential(
            CBRelu(in_channels, inter_channels, 7, 2, 3),
            CBRelu(inter_channels, inter_channels, 3, 2, 1),
            CBRelu(inter_channels, inter_channels, 1, 2, 0))      

        # self.flattens = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(flatten_shape, inter_features)
        # )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.ReLU(),
            nn.Linear(inter_features, num_class))

    def forward(self, input):
        """Forward"""
        repr = self.transform(input)
        # print("[spatial]  repr:\t",repr.shape)
        # flats = self.flattens(repr)
        # print("[spatial]  flats:\t",flats.shape)
        # return repr, logits
        # logits = self.classifier(flats)
        # print("[spatial] logits:\t",logits.shape)
        return repr, None

class Skips(nn.Module):
    def __init__(self, cfg, input, id):
        super().__init__()
        # Common configs
        self.cfg = cfg
        self.match_fc = cfg['__fc_features__']
        self.inter_fc = cfg['skips']['inter_ch']
        batch, channels, width, height = input.shape
       
        self.skips = nn.Sequential(
            CBRelu(channels, self.match_fc, 1, 1, 0),
            nn.AdaptiveAvgPool2d(1))

    def forward(self, input):
        x = self.skips(input)
        return x

class FeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1):
        """feature fusion module"""
        super().__init__()
        self.conv1 = CBRelu(in_channels, out_channels, 1, 1, 0)
        self.atten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            CBRelu(out_channels, out_channels, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        """forward"""
        x = torch.cat([x1, x2], dim=1)
        out = self.conv1(x)
        att = self.atten(out)
        out = out + out * att
        return out

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

        # Create Tail Layers
        self.skip_layers.append(Skips(cfg=self.cfg, input=input, id=id))
        # self.tail_layer = self.exactly[-1]
        self.lazy_num += 1
        print('     || ')
        print("<tail layer>")
        print('---------------------------------------------------')
        print("Model Set Complete!")
        
        self.tail_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.cfg['__fc_features__'], self.cfg['num_class'], bias=True)
        )
        input = self.tail_layers(input)
        return input

    def forward(self, input):
        print("[init] input:\t",input.shape)
        # repr, logits = self.spatial(input)

        x = self.head_layer(input)
        print('[head]x.shape:\t',x.shape)
        for i, (feat, skip) in enumerate(
            zip(self.feats_layers, self.skip_layers)):
            print('[feat#'+str(i)+']pre-x:\t',x.shape)
            x = feat(x)
            print('[feat#'+str(i)+']post-x:\t',x.shape)
            val = skip(x)

        for fetc in self.fetc_layers:
            print('[fetc]pre-x:\t',x.shape)
            x = fetc(x)
            print('[fetc]post-x:\t',x.shape)

        val = self.tail_layers(x)
        print('[tail]val.shape:\t',val.shape)

        return val  

def main():
    """Main function"""
    print("Lazyfy")
    args = parse_args()
    cfg = read_yaml(args.configs)
    model = LazyNet(cfg)
    
    x = torch.rand((1,3,256,256))
    result = model(x)
    # print(model)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()

###https://madebyollin.github.io/convnet-calculator/