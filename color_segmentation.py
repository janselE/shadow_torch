import random
import torch

class Color_Mask:
    def __init__(self, output):
        self.output = output

        self.red   = []
        self.green = []
        self.blue  = []

        for i in range(0, self.output):
            self.red.append(random.randint(0, 255))
            self.green.append(random.randint(0, 255))
            self.blue.append(random.randint(0, 255))

    def add_color(self, tensor):
        tensor = tensor.clone().squeeze()
        segmentation = torch.zeros(3, tensor.shape[0], tensor.shape[1]).to(torch.uint8)

        for category in range(0, self.output):
            if category == 0:
                segmentation[0, tensor == category] = 0
                segmentation[1, tensor == category] = 0
                segmentation[2, tensor == category] = 0

            else:
                segmentation[0, tensor == category] = self.red[category]
                segmentation[1, tensor == category] = self.green[category]
                segmentation[2, tensor == category] = self.blue[category]

        return segmentation


