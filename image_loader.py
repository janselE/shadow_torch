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

# Create the data class, this is done to load the data into the pytorch model
# this class might be slow because is reading the image at the time is being requested
# in this manner we do not load a lot of the images and run out of memmory
class ShadowDataset(Dataset):
    # Constructor
<<<<<<< HEAD
    def __init__(self, h, w, transform=None, use_random_scale=False, use_random_affine=True):
=======
    def __init__(self, h, w, use_random_scale=False, use_random_affine=False):
>>>>>>> 29a183b75135b1542200af6344e733391d1e0daf
        self.imgs = read_dataset('./ISTD_Dataset/train/train_A/*.png')  # known name, this is for local
        #self.len = 20
        self.len = len(self.imgs)  # read all the images of the dataset
        # self.transform = transform
        self.h = h
        self.w = w
<<<<<<< HEAD
=======
        # self.size = int(self.len/2)
>>>>>>> 29a183b75135b1542200af6344e733391d1e0daf

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
        #image = transforms.RandomCrop((self.h, self.w))(image)
        image = np.asarray(image).astype(np.float32)
        target = image

        if self.use_random_scale:
            scale_factor = (np.random.rand() * (self.scale_max - self.scale_min)) + \
                    self.scale_min
            image = cv2.resize(image, dsize=None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        img, coords = pad_and_or_crop(image, self.input_sz, mode="random")

        #img_ir = img[:, :, 3] # we do not have an infrared channel
        #img = img[:, :, :3]

        img1 = Image.fromarray(img.astype(np.uint8))
        img2 = self.jitter_tf(img1)

        img1 = np.asarray(img1)
        img2 = np.asarray(img2)

        # skiped the sobel part

        img1 = img1.astype(np.float32) / 255
        img2 = img2.astype(np.float32) / 255

        img1 = torch.from_numpy(img1).permute(2, 0, 1).cuda()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).cuda()

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



class ShadowShadowFreeDataset(Dataset):
    # Constructor
    def __init__(self, h, w, use_random_scale=False, use_random_affine=False):
        self.imgs_s = read_dataset('./ISTD_Dataset/train/train_A/*.png')  # shadow containing images
        self.imgs_sf = read_dataset('./ISTD_Dataset/train/train_C/*.png')  # shadow-free images
        #self.len = 20
        self.len = len(self.imgs_s)  # read all the images of the dataset
        # self.transform = transform
        self.h = h
        self.w = w
        # self.size = int(self.len/2)

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

        # Are images in same order? Need to check
        image = Image.open(self.imgs_s[index])
        image = np.asarray(image).astype(np.float32)
        sf_image = Image.open(self.imgs_s[index])
        sf_image = np.asarray(sf_image).astype(np.float32)

        if self.use_random_scale:
            scale_factor = (np.random.rand() * (self.scale_max - self.scale_min)) + \
                    self.scale_min
            image = cv2.resize(image, dsize=None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            sf_image = cv2.resize(image, dsize=None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        img, coords = pad_and_or_crop(image, self.input_sz, mode="random")
        sf_img, _ = pad_and_or_crop(sf_image, self.input_sz, mode="random")

        img1 = Image.fromarray(img.astype(np.uint8))
        sf_img = Image.fromarray(sf_img.astype(np.uint8))
        img2 = self.jitter_tf(img1)

        img1 = np.asarray(img1)
        sf_img = np.asarray(sf_img)
        img2 = np.asarray(img2)

        # skiped the sobel part

        img1 = img1.astype(np.float32) / 255
        sf_img = sf_img.astype(np.float32) / 255 * 2 - 2  # scales to [-1, 1] for tanh output
        img2 = img2.astype(np.float32) / 255

        img1 = torch.from_numpy(img1).permute(2, 0, 1).cuda()
        sf_img = torch.from_numpy(sf_img).permute(2, 0, 1).cuda()  # not sure if this is needed, but shouldn't matter
        img2 = torch.from_numpy(img2).permute(2, 0, 1).cuda()

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

        return img1, img2, affine2_to_1, mask_img1, sf_img

    # Get items
    def __len__(self):
        return self.len

