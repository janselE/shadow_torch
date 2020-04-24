from net10a import SegmentationNet10aHead, SegmentationNet10aTrunk, SegmentationNet10a
from vgg import *


class SegmentationNet10aTwoHead(VGGNet):
    def __init__(self):
        super(SegmentationNet10aTwoHead, self).__init__()
        # this variable was supposed to ba a static variable
        # from SegmentationNet10a
        self.cfg = [(64, 1), (128, 1), ('M', None), (256, 1), (256, 1), (512, 2), (512, 2)]  # 30x30 recep field
        output = 2

        #self.batchnorm_track = config.batchnorm_track
        self.batchnorm_track = False

        self.trunk  = SegmentationNet10aTrunk(cfg=self.cfg)
        self.head_A = SegmentationNet10aHead(output_k=output, cfg=self.cfg)
        self.head_B = SegmentationNet10aHead(output_k=output, cfg=self.cfg)

        self._initialize_weights()

    def forward(self, x, head="B"):
        x = self.trunk(x)
        if head == "A":
            x = self.head_A(x)
        elif head == "B":
            x = self.head_B(x)
        else:
            assert (False)

        return x




