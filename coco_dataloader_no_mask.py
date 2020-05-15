import glob
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import json
import yaml


import torchvision.transforms as tf
import torchvision.transforms.functional as TF
import random

from transforms import *

from PIL import Image, ImageDraw
#from pycocotools.coco import COCO
import os

import time
from threading import Thread


# take as parameter the filename that we already know
# we are just changing the part of the name that is different
# this method has to be changed for different datasets
def read_dataset(filename):
    imgs_names = []
    for filename in glob.glob(filename):
        imgs_names.append(filename)
    return imgs_names

class CocoDataloader(torch.utils.data.Dataset):
    # Constructor
    def __init__(self, h, w, use_random_scale=False, use_random_affine=True):
        self.imgs, _, _ = read_dataset(path)  # known name, this is for local
        self.len = len(self.imgs)  # read all the images of the dataset
        self.h = h
        self.w = w

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

        # Getter
    def __getitem__(self, index):
        # baseline transformations

        image = Image.open(self.imgs[index])
        image = np.asarray(image).astype(np.float32)
        target = image

        if self.use_random_scale:
            scale_factor = (np.random.rand() * (self.scale_max - self.scale_min)) + \
                    self.scale_min
            image = cv2.resize(image, dsize=None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        img, coords = pad_and_or_crop(image, self.input_sz, mode="random")

        img1 = Image.fromarray(img.astype(np.uint8))
        img2 = self.jitter_tf(img1)

        img1 = np.asarray(img1)
        img2 = np.asarray(img2)

        # skiped the sobel part

        img1 = img1.astype(np.float32) / 255
        img2 = img2.astype(np.float32) / 255

        img1 = torch.from_numpy(img1).permute(2, 0, 1)
        img2 = torch.from_numpy(img2).permute(2, 0, 1)

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

        mask_img1 = torch.ones(self.input_sz, self.input_sz).to(torch.uint8) #cuda

        return img1, img2, affine2_to_1, mask_img1

    # Get items
    def __len__(self):
        return self.len
