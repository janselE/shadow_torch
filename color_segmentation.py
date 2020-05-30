import random
import torch

class Color_Mask:
    def __init__(self, output):
        self.output = output

        self.red   = []
        self.green = []
        self.blue  = []

        for i in range(0, self.output):
            self.red.append(random.randint(0, 256))
            self.green.append(random.randint(0, 256))
            self.blue.append(random.randint(0, 256))

    def add_color(self, tensor):
        width = tensor.shape[0]
        heigth = tensor.shape[1]
        channels = 3
        segmentation = torch.zeros(width, heigth, 3).to(torch.uint8) # this might be an error becuase the way im giving the shape

        for category in range(0, self.output):
            segmentation[tensor == category, 0] = self.red[category]
            segmentation[tensor == category, 1] = self.green[category]
            segmentation[tensor == category, 2] = self.blue[category]

        return segmentation


