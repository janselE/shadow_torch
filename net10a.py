
import torch.nn as nn
import torch.nn.functional as F

from vgg import VGGTrunk, VGGNet

__all__ = ["SegmentationNet10a"]


class SegmentationNet10aTrunk(VGGTrunk):
    def __init__(self, cfg):

        self.conv_size = 3
        self.pad = 1
        self.cfg = cfg
        self.in_channels = 4 #config.in_channels if hasattr(config, 'in_channels') else 3

        # on the self parameter we send all of this
        # to the super class
        super(SegmentationNet10aTrunk, self).__init__()

        self.features = self.layers

    def forward(self, x):
        x = self.features(x)  # do not flatten
        return x

class SegmentationNet10aHead(nn.Module):
    def __init__(self, input_sz, output_k, cfg, num_sub_heads):
        super(SegmentationNet10aHead, self).__init__()

        #self.batchnorm_track = config.batchnorm_track
        self.batchnorm_track = False

        self.cfg = cfg
        num_features = self.cfg[-1][0]

        self.num_sub_heads = num_sub_heads

        self.heads = nn.ModuleList([nn.Sequential(nn.Conv2d(num_features, output_k, kernel_size=1,
                                                            stride=1, dilation=1, padding=1, bias=False),
                                                            nn.Softmax2d()) for _ in range(self.num_sub_heads)])

        self.input_sz = input_sz #config.input_sz # this is the image size, not sure about this

    def forward(self, x):
        results = []
        for i in range(self.num_sub_heads):
            x_i = self.heads[i](x)
            x_i = F.interpolate(x_i, size=self.input_sz, mode="bilinear")
            results.append(x_i)

        return results

class SegmentationNet10a(VGGNet):
    def __init__(self, num_sub_heads, input_sz, output_k):
        super(SegmentationNet10a, self).__init__()

        # this variable was supposed to be used as a static var
        # here they defined the structure of the network
        self.cfg = [(64, 1), (128, 1), ('M', None), (256, 1), (256, 1), (512, 2), (512, output_k)]  # 30x30 recep field

        #self.batchnorm_track = config.batchnorm_track
        self.batchnorm_track = False

        self.trunk = SegmentationNet10aTrunk(cfg=self.cfg)
        self.head = SegmentationNet10aHead(input_sz=input_sz,
                                           output_k=output_k,
                                           cfg=self.cfg,
                                           num_sub_heads=num_sub_heads)

        self._initialize_weights()

    def forward(self, x):
        x = self.trunk(x)
        x = self.head(x)
        return x

