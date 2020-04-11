from network import *
from image_loader import *


trainloader = DataLoader(dataset=Data(), batch_size=16) # Create the loader for the model

for x, y in trainloader:
    print(x.shape)

