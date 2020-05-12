
import torch.nn as nn
import torch.nn.functional as F

from vgg import VGGTrunk, VGGNet

__all__ = ["SegmentationNet10a"]


class SegmentationNet10aTrunk(VGGTrunk):
    def __init__(self, cfg):
        super(SegmentationNet10aTrunk, self).__init__()

        #self.batchnorm_track = config.batchnorm_track

        #assert (config.input_sz % 2 == 0)

        self.conv_size = 3
        self.pad = 1
        self.cfg = cfg
        self.in_channels = 3 #config.in_channels if hasattr(config, 'in_channels') else 3

        self.features = self.layers #self._make_layers()

    def forward(self, x):
        x = self.features(x)  # do not flatten
        return x

class SegmentationNet10aHead(nn.Module):
    def __init__(self, output_k, cfg, num_sub_heads):
        super(SegmentationNet10aHead, self).__init__()

        #self.batchnorm_track = config.batchnorm_track
        self.batchnorm_track = False

        self.cfg = cfg
        num_features = self.cfg[-1][0]

        self.num_sub_heads = num_sub_heads #config.num_sub_heads

        self.heads = nn.ModuleList([nn.Sequential(nn.Conv2d(num_features, output_k, kernel_size=1,
                                                            stride=1, dilation=1, padding=1, bias=False),
                                                            nn.Softmax2d()) for _ in range(self.num_sub_heads)])

        self.input_sz = 240 #config.input_sz # this is the image size, not sure about this

    def forward(self, x):
        results = []
        for i in range(self.num_sub_heads):
            x_i = self.heads[i](x)
            x_i = F.interpolate(x_i, size=self.input_sz, mode="bilinear")
            results.append(x_i)

        return results

class SegmentationNet10a(VGGNet):

    def __init__(self, num_sub_heads, output_k):
        super(SegmentationNet10a, self).__init__()
        # this variable was supposed to be used as a static var
        self.cfg = [(64, 1), (128, 1), ('M', None), (256, 1), (256, 1), (512, 2), (512, 12)]  # 30x30 recep field

        #self.batchnorm_track = config.batchnorm_track
        self.batchnorm_track = False

        self.trunk = SegmentationNet10aTrunk(cfg=self.cfg)
        self.head = SegmentationNet10aHead(output_k=output_k, # this is for the number of classes
                                           cfg=self.cfg,
                                           num_sub_heads=num_sub_heads)

        self._initialize_weights()

    def forward(self, x):
        x = self.trunk(x)
        x = self.head(x)
        return x

