import torchvision
import torchvision.transforms as transforms
import torch
from image_loader import *
import matplotlib.pyplot as plt


h, w = 240, 240
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()])

dataloader = DataLoader(dataset=Data(h, w, transform), batch_size=32, shuffle=True)  # Create the loader for the model

for i, data in enumerate(dataloader, 0):
    original_image, transformed = data
    #print(original_image.shape, transformed.shape)
    print(original_image.shape)

    t = original_image[0].numpy().reshape(h, w, 3)
    t2 = transformed[0].numpy().reshape(h, w, 3)
    im = transforms.ToPILImage()(original_image[0])
    im2 = transforms.ToPILImage()(transformed[0])
    im2.save('test.png')

    #plt.figure(i)
    #plt.imshow(im)
    #plt.figure(i+1)
    #plt.imshow(im2)
    ##plt.figure(i+2)
    ##plt.imshow(t3)
    #plt.show()

