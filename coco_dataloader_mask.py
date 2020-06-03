import glob
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import json
import yaml

import cv2


import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

from transforms import *

from PIL import Image, ImageDraw
#from pycocotools.coco import COCO
import os.path as os
import yaml

root = 'data3'

# take as parameter the filename that we already know
# we are just changing the part of the name that is different
# this method has to be changed for different datasets
def read_dataset(filename):
    imgs_names = []
    for fn in glob.glob(filename):
        imgs_names.append(fn)

    return imgs_names


def custom_greyscale_numpy(img, include_rgb=True):
    # Takes and returns a channel-last numpy array, uint8

    # use channels last for cvtColor
    h, w, c = img.shape
    grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(h, w,
                                                             1)  # new memory
    if include_rgb:
      img = np.concatenate([img, grey_img], axis=2)
    else:
      img = grey_img

    return img

# this is to generate the new labels
_sorted_coarse_names = [
        "electronic-things",  # 0
        "appliance-things",  # 1
        "food-things",  # 2
        "furniture-things",  # 3
        "indoor-things",  # 4
        "kitchen-things",  # 5
        "accessory-things",  # 6
        "animal-things",  # 7
        "outdoor-things",  # 8
        "person-things",  # 9
        "sports-things",  # 10
        "vehicle-things",  # 11

        "ceiling-stuff",  # 12
        "floor-stuff",  # 13
        "food-stuff",  # 14
        "furniture-stuff",  # 15
        "rawmaterial-stuff",  # 16
        "textile-stuff",  # 17
        "wall-stuff",  # 18
        "window-stuff",  # 19
        "building-stuff",  # 20
        "ground-stuff",  # 21
        "plant-stuff",  # 22
        "sky-stuff",  # 23
        "solid-stuff",  # 24
        "structural-stuff",  # 25
        "water-stuff"  # 26
        ]

_sorted_coarse_name_to_coarse_index = \
  {n: i for i, n in enumerate(_sorted_coarse_names)}

def _find_parent(name, d):
    for k, v in d.items():
        if isinstance(v, list):
              if name in v:
                    yield k
        else:
            assert (isinstance(v, dict))
            for res in _find_parent(name, v):  # if it returns anything to us
                yield res

def generate_fine_to_coarse():
    fine_index_to_coarse_index = {}
    fine_name_to_coarse_name = {}

    with open(root + "/cocostuff_fine_raw.txt") as f:
        l = [tuple(pair.rstrip().split('\t')) for pair in f]
        l = [(int(ind), name) for ind, name in l]

    with open(root + "/cocostuff_hierarchy.y") as f:
        d = yaml.load(f, Loader=yaml.FullLoader)

    for fine_ind, fine_name in l:
        assert (fine_ind >= 0 and fine_ind < 182)
        parent_name = list(_find_parent(fine_name, d))
        # print("parent_name of %d %s: %s"% (fine_ind, fine_name, parent_name))
        assert (len(parent_name) == 1)
        parent_name = parent_name[0]
        parent_ind = _sorted_coarse_name_to_coarse_index[parent_name]
        assert (parent_ind >= 0 and parent_ind < 27)

        fine_index_to_coarse_index[fine_ind] = parent_ind
        fine_name_to_coarse_name[fine_name] = parent_name

    assert (len(fine_index_to_coarse_index) == 182)

    return fine_index_to_coarse_index, fine_name_to_coarse_name

class CocoDataloader(torch.utils.data.Dataset):
    # Constructor
    def __init__(self, h,  use_random_scale=False, use_random_affine=True, mode="train"):
        if mode == "train":
            # this is for the normal images
            path = os.join(root, "images", "train2017")
            path = path + '/*.jpg'
            self.imgs = read_dataset(path)

            path = os.join(root, "annotations", "train2017")
            path = path + '/*.png'
            self.mask = read_dataset(path)
        else:
            # this is for the normal images
            path = os.join(root, "images", "val2017")
            path = path + '/*.jpg'
            self.imgs = read_dataset(path)

            path = os.join(root, "annotations", "val2017")
            path = path + '/*.png'
            self.mask = read_dataset(path)

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

        self._fine_to_coarse_index, self.fine_tor_coarse_name = generate_fine_to_coarse()


    def __getitem__(self, index):
        img = cv2.imread(self.imgs[index], cv2.IMREAD_COLOR).astype(np.uint8)
        labels = cv2.imread(self.imgs[index], cv2.IMREAD_GRAYSCALE).astype(np.int32)


        labels[labels == 255] = -1

        img, coords = pad_and_or_crop(img, self.input_sz, mode="random")
        labels, _ = pad_and_or_crop(labels, self.input_sz, mode="fixed",coords=coords)

        new_labels = np.zeros(labels.shape, dtype=labels.dtype)
        for c in range(0, 182):
           new_labels[labels == c] = self._fine_to_coarse_index[c]
        labels = new_labels


        first_allowed_index = 12
        mask_img = (labels >= first_allowed_index)

        labels = torch.from_numpy(new_labels)

        mask_img = torch.from_numpy(mask_img.astype(np.uint8))

        img = Image.fromarray(img.astype(np.uint8))
        img2 = self.jitter_tf(img)

        img1 = np.array(img)
        img2 = np.array(img2)

        img1 = custom_greyscale_numpy(img1, True)
        img2 = custom_greyscale_numpy(img2, True)

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

#        print(img1.shape)
#        print(img2.shape)
#        print(affine2_to_1.shape)
#        print(mask_img.shape)
#        print(labels.shape)
#        print(labels)

        return img1, img2, affine2_to_1, mask_img, labels




#        img1 = np.asarray(img1)
#        mask = np.asarray(mask)
#        img2 = np.asarray(img2)
#
#        img1 = img1.astype(np.float32) / 255
#        img2 = img2.astype(np.float32) / 255
#        mask = mask.astype(np.float32)
#
#
#        img1 = torch.from_numpy(img1).permute(2, 0, 1)
#        img2 = torch.from_numpy(img2).permute(2, 0, 1)
#        mask_cat = torch.zeros(1, self.input_sz, self.input_sz).to(torch.uint8)
#
#        mask = mask.reshape(1, self.input_sz, self.input_sz)
#
#        # not the best way but this is to flip the labels
#        mask_cat[0] = Variable(torch.Tensor(mask))
#
#        if self.use_random_affine:
#            affine_kwargs = {"min_rot": self.aff_min_rot, "max_rot": self.aff_max_rot,
#                    "min_shear": self.aff_min_shear,
#                    "max_shear": self.aff_max_shear,
#                    "min_scale": self.aff_min_scale,
#                    "max_scale": self.aff_max_scale}
#
#
#            img2, affine1_to_2, affine2_to_1 = random_affine(img2, **affine_kwargs)  #
#
#        else:
#            affine2_to_1 = torch.zeros([2, 3]).to(torch.float32) # cuda
#            affine2_to_1[0, 0] = 1
#            affine2_to_1[1, 1] = 1
#
#        if np.random.rand() > self.flip_p:
#            img2 = torch.flip(img2, dims=[2])
#            affine2_to_1[0, :] *= -1
#
#        mask_cat = mask_cat.view(1, self.input_sz, self.input_sz)
#        mask_img1 = torch.ones(self.input_sz, self.input_sz).to(torch.uint8) #cuda
#
#
#        return img1, img2, affine2_to_1, mask_img1, mask_cat

    def __len__(self):
        return len(self.mask)
