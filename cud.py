from __future__ import print_function
import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())

x = torch.rand(5, 3)
print(x)
