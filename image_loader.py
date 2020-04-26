import glob
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import scipy.io as sio

from transforms import *

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
    def __init__(self, h, w, transform=None, use_random_scale=False):
        self.imgs = read_dataset('../ISTD_Dataset/train/train_A/*.png') # known name, this is for local
        self.len = 20
        #self.len = len(self.imgs) # read all the images of the dataset
        self.transform = transform
        self.h = h
        self.w = w
        self.size = int(self.len/2)

        # config parameters
        self.use_random_scale = use_random_scale
        self.scale_max = 1.4
        self.scale_min = 0.6
        self.input_sz = h

    # Getter
    def __getitem__(self, index):
        # baseline transformations

        image = sio.loadmat(self.imgs[index])["img"]
        #image = transforms.RandomCrop((self.h, self.w))(image)
        image = np.asarray(image).astype(np.float32)
        target = image

        if self.use_random_scale:
            scale_factor = (np.random.rand() * (self.scale_max - self.scale_min)) + \
                    self.scale_min
            image = cv2.resize(image, dsize=None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        img, coords = pad_and_or_crop(image, self.input_sz, mode="random")

        img_ir = img[:, :, 3]
        img = img[:, :, :3]

        img1 = Image.fromarray(img.astype(np.uint8))

        #if self.transform:
        #    target = self.transform(image)

        self.x = transforms.ToTensor()(image)
        self.y = transforms.ToTensor()(target)

        return self.x, self.y

    # Get items
    def __len__(self):
        return self.size


