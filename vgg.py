import torch.nn as nn

class VGGTrunk(nn.Module):
    def __init__(self):
        super(VGGTrunk, self).__init__()
        # fix this later
        #self.cfg = [(64, 1), (128, 1), ('M', None), (256, 1), (256, 1), (512, 2), (512, 12)]  # 30x30 recep field
        #self.conv_size = 3
        #self.pad = 1
        #self.in_channels = 4 #config.in_channels if hasattr(config, 'in_channels') else 3
        self.batchnorm_track = False

        layers = self._make_layers()
        self.layers = nn.Sequential(*layers)

    def _make_layers(self, batch_norm=True):
        layers = []
        in_channels = 4 #self.in_channels
        for tup in self.cfg:
            assert (len(tup) == 2)

            out, dilation = tup
            sz = self.conv_size
            stride = 1
            pad = self.pad  # to avoid shrinking

            if out == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif out == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, out, kernel_size=sz, stride=stride, padding=pad,
                                   dilation=dilation, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(out, track_running_stats=self.batchnorm_track),
                            nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = out

        return layers

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.batchnorm_track = False

    def _initialize_weights(self, mode='fan_in'):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                assert (m.track_running_stats == self.batchnorm_track)
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

