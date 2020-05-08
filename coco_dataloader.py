import glob
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import json


import torchvision.transforms as tf
import torchvision.transforms.functional as TF
import random

from transforms import *

from PIL import Image, ImageDraw
#from pycocotools.coco import COCO
import os

import time
from threading import Thread

class CocoDataloader(torch.utils.data.Dataset):
    samples = []
    classes = []
    def __init__(self, root, annotation, input_sz, use_random_scale=False, use_random_affine=True, transforms=None, classes_path=None):
        self.root = root
        self.transforms = transforms

        x = []
        if classes_path != None:
            with open(classes_path,'r') as f:
                x = f.readlines()
            CocoDataloader.classes = [item.replace('\n', '.jpg') for item in x]

        with open(annotation) as f:
            self.data = json.load(f)

        self.jitter_tf = tf.ColorJitter(brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.125)

        self.aff_min_rot = 10.  # not sure value to use
        self.aff_max_rot = 10.  # not sure value to use
        self.aff_min_shear = 10.
        self.aff_max_shear = 10.
        self.aff_min_scale = 0.8
        self.aff_max_scale = 1.2
        self.flip_p = 0.5
        self.use_random_affine = use_random_affine
        self.use_random_scale = use_random_scale

        self.input_sz = input_sz

    def getImgAndMask(self, index, w, h):
        img_id = self.data['images'][index]
        len_ = len(self.data['annotations'])
        ann_ids = []
        for i in range(len_):
            if self.data['annotations'][i]['image_id'] == self.data['images'][index]['id']:
                ann_ids.append(i)
        path = img_id['file_name']

        if path not in CocoDataloader.classes and len(CocoDataloader.classes) > 0:
            return None, None

        img = Image.open(os.path.join(self.root, path))
        img = img.convert('RGB')
        num_objs = len(ann_ids)
        w, h = img.size

        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        for i in range(num_objs):
            seg = self.data['annotations'][ann_ids[i]]['segmentation']
            cat = self.data['annotations'][ann_ids[i]]['category_id']
            crowd = self.data['annotations'][ann_ids[i]]['iscrowd']

            if crowd == 1:
                return None, None

            for n in range(len(seg)):
                draw.polygon(seg[n], outline=None, fill=cat)
        del draw

        return img, mask

    def __getitem__(self, index):
        img, mask = self.getImgAndMask(index, self.input_sz, self.input_sz)

        if img is None:
            return None


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

        #img1 = img1.view(3, self.input_sz, self.input_sz)
        #img2 = img2.view(3, self.input_sz, self.input_sz)
        mask_cat = mask_cat.view(1, self.input_sz, self.input_sz)
        mask_img1 = torch.ones(self.input_sz, self.input_sz).to(torch.uint8) #cuda


        return img1, img2, affine2_to_1, mask_img1, mask_cat

    def __len__(self):
        return len(self.data['images'])

    def collate_fn(batch):
        #classes = [0, 1, 2] # these are the selected classes, we can modify which ones we want
        len_batch = len(batch)

        #batch = list(filter(lambda x:x is not None, batch))

        # this create a new batch of the good samples
        # the samples are filtered if they are having a bad
        # segmentation description and if they are not
        # part of the classes that we are interested
        new_batch = []
        for b in batch:
            if b is not None: #and torch.max(b[4]) in classes: # the amount of classes that we want
                new_batch.append(b)
        #print("s ", len(new_batch))

        if len_batch > len(new_batch):
            diff = len_batch - len(new_batch)

            for i in range(diff):
                rand = 0
                if len(new_batch) == 0:
                    rand = random.randint(0, abs(len(CocoDataloader.samples) - 1))
                    samp = CocoDataloader.samples[rand]
                    new_batch.append(samp)

                else:
                    rand = random.randint(0, abs(len(new_batch) - 1))
                    samp = new_batch[rand]
                    new_batch.append(samp)

                if np.random.rand() < 0.5 and len(CocoDataloader.samples) < 32 or len(CocoDataloader.samples) == 0: # this is just a number to limitate the memory usage
                    CocoDataloader.samples.append(samp)

                #print("s ", len(new_batch)," r ", rand)

        return torch.utils.data.dataloader.default_collate(new_batch)

