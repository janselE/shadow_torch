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
        segmentation = torch.zeros(tensor.shape[0], tensor.shape[1], 3).to(torch.uint8) # this might be an error becuase the way im giving the shape
        print(segmentation.shape)
        print(tensor.shape)
        exit()

        for category in range(0, self.output):
            segmentation[0, tensor == category] = self.red[category]
            segmentation[1, tensor == category] = self.green[category]
            segmentation[2, tensor == category] = self.blue[category]

        return segmentation


