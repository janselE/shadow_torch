import glob
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

from PIL import Image


# take as parameter the filename that we already know
# we are just changing the part of the name that is different
# this method has to be changed for different datasets
def read_dataset(filename):
    imgs_names = []
    for filename in glob.glob(filename):
        imgs_names.append(filename)

    return imgs_names

#def reader(filename):
#    imgs_c = []
#
#
#    imgs_names = read_dataset(filename)
#    amt = 20
#
#    # Loop that read in the images and the target images
#    for i in range(0, amt):
#        imgs_c.append(cv2.imread(imgs_names[i]))
#
#    imgs_c = np.asarray(imgs_c)
#
#
#    return imgs_c

# Create the data class, this is done to load the data into the pytorch model
# this class might be slow because is reading the image at the time is being requested
# in this manner we do not load a lot of the images and run out of memmory
class Data(Dataset):
    # Constructor
    def __init__(self, h, w, mode="A", transform=None):
        self.mode = mode
        self.imgs = read_dataset('../ISTD_Dataset/train/train_A/*.png') # known name, this is for local
        self.len = 20
        #self.len = len(self.imgs) # read all the images of the dataset
        self.transform = transform
        self.h = h
        self.w = w
        self.size = int(self.len/2)

    # Getter
    def __getitem__(self, index):
        # baseline transformations
        if "B" == self.mode:
            index = index + self.size

        image = Image.open(self.imgs[index])
        image = transforms.RandomCrop((self.h, self.w))(image)
        target = image

        if self.transform:
            target = self.transform(image)

        self.x = transforms.ToTensor()(image)
        self.y = transforms.ToTensor()(target)

        return self.x, self.y

    # Get items
    def __len__(self):
        return self.size


