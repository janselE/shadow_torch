import glob
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import json
import yaml


import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

from transforms import *

from PIL import Image, ImageDraw
#from pycocotools.coco import COCO
import os

import time
from threading import Thread


#path = '/Users/janselherrera/Documents/Projects/Research/Shadow/compressed/train2017_mask/*.jpg' # mac
#path = '/home/jansel/Documents/Research/coco_dataset/data/train2017_mask/*.jpg' # msi
path_train = 'data/train2017_mask/*.jpg' # server
path_val = 'data/val2017_mask/*.jpg' # server


# take as parameter the filename that we already know
# we are just changing the part of the name that is different
# this method has to be changed for different datasets
def read_dataset(filename):
    imgs_names = []
    mask_names = []
    for filename in glob.glob(filename):
        mask_names.append(filename)
        imgs_names.append(filename.replace("_mask", ""))

    return imgs_names, mask_names

class CocoDataloader(torch.utils.data.Dataset):
    # Constructor
    def __init__(self, h,  use_random_scale=False, use_random_affine=True, mode="train"):
        if mode == "train":
            self.imgs, self.mask = read_dataset(path_train)  # known name, this is for local
        else:
            self.imgs, self.mask = read_dataset(path_val)  # known name, this is for local

        self.len_ = len(self.imgs)  # read all the images of the dataset
        self.h = h
        self.w = h

        # config parameters
        self.use_random_scale = use_random_scale
        self.use_random_affine = use_random_affine
        self.scale_max = 1.4
        self.scale_min = 0.6
        self.input_sz = h
        self.aff_min_rot = 10.  # not sure value to use
        self.aff_max_rot = 10.  # not sure value to use
        self.aff_min_shear = 10.
        self.aff_max_shear = 10.
        self.aff_min_scale = 0.8
        self.aff_max_scale = 1.2
        self.flip_p = 0.5

        self.jitter_tf = transforms.ColorJitter(brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.125)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index])
        img = img.convert('RGB')
        mask = Image.open(self.mask[index])

        image = np.asarray(img).astype(np.float32)
        mask = np.asarray(mask).astype(np.float32)

        img, coords = pad_and_or_crop(image, self.input_sz, mode="random")
        mask, _ = pad_and_or_crop(mask, self.input_sz, mode="fixed",coords=coords)

        img1 = Image.fromarray(img.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))
        img2 = self.jitter_tf(img1)

        img1 = np.asarray(img1)
        mask = np.asarray(mask)
        img2 = np.asarray(img2)

        img1 = img1.astype(np.float32) / 255
        mask = mask.astype(np.float32)
        img2 = img2.astype(np.float32) / 255


        img1 = torch.from_numpy(img1).permute(2, 0, 1)
        img2 = torch.from_numpy(img2).permute(2, 0, 1)
        mask_cat = torch.zeros(1, self.input_sz, self.input_sz).to(torch.uint8)

        mask = mask.reshape(1, self.input_sz, self.input_sz)

        # not the best way but this is to flip the labels
        mask_cat[0] = Variable(torch.Tensor(mask))

        if self.use_random_affine:
            affine_kwargs = {"min_rot": self.aff_min_rot, "max_rot": self.aff_max_rot,
                    "min_shear": self.aff_min_shear,
                    "max_shear": self.aff_max_shear,
                    "min_scale": self.aff_min_scale,
                    "max_scale": self.aff_max_scale}


            img2, affine1_to_2, affine2_to_1 = random_affine(img2, **affine_kwargs)  #

        else:
            affine2_to_1 = torch.zeros([2, 3]).to(torch.float32) # cuda
            affine2_to_1[0, 0] = 1
            affine2_to_1[1, 1] = 1

        if np.random.rand() > self.flip_p:
            img2 = torch.flip(img2, dims=[2])
            affine2_to_1[0, :] *= -1

        mask_cat = mask_cat.view(1, self.input_sz, self.input_sz)
        mask_img1 = torch.ones(self.input_sz, self.input_sz).to(torch.uint8) #cuda


        print(type(img1))
        print(type(img2))

        print(type(mask_cat))
        print(mask_cat)
        print(torch.max(mask_cat))
        exit()

        return img1, img2, affine2_to_1, mask_img1, mask_cat

    def __len__(self):
        return len(self.mask)
