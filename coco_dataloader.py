import glob
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


import torchvision.transforms as tf
import torchvision.transforms.functional as TF
import random

from transforms import *

from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import os

import time
from threading import Thread

class CocoDataloader(torch.utils.data.Dataset):
    samples = []
    def __init__(self, root, annotation, input_sz, use_random_scale=False, use_random_affine=True, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

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

    def getImg(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path))

        return img, coco_annotation

    def getImgAndMask(self, index, w, h):
        img, coco_annotation = self.getImg(index)
        img = img.convert('RGB')

        # number of objects in the image
        num_objs = len(coco_annotation)

        w, h = img.size

        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        for i in range(num_objs):
            seg = coco_annotation[i]['segmentation']
            cat = coco_annotation[i]['category_id']
            crowd = coco_annotation[i]['iscrowd']
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


        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
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

        img1 = img1.view(3, self.input_sz, self.input_sz)
        img2 = img2.view(3, self.input_sz, self.input_sz)
        mask_cat = mask_cat.view(1, self.input_sz, self.input_sz)
        mask_img1 = torch.ones(self.input_sz, self.input_sz).to(torch.uint8) #cuda


        return img1, img2, affine2_to_1, mask_img1, mask_cat

    def __len__(self):
        return len(self.ids)

    def collate_fn(batch):
        classes = [0, 1, 2] # these are the selected classes, we can modify which ones we want
        len_batch = len(batch)

        #batch = list(filter(lambda x:x is not None, batch))

        # this create a new batch of the good samples
        # the samples are filtered if they are having a bad
        # segmentation description and if they are not
        # part of the classes that we are interested
        new_batch = []
        for b in batch:
            if b is not None and torch.max(b[4]) in classes: # the amount of classes that we want
                new_batch.append(b)
        print("s ", len(new_batch))

        if len_batch > len(new_batch):
            diff = len_batch - len(new_batch)

            for i in range(diff):
                rand = 0
                if len(new_batch) == 0:
                    rand = random.randint(0, len(CocoDataloader.samples) - 1)
                    samp = CocoDataloader.samples[rand]
                    new_batch.append(samp)

                else:
                    rand = random.randint(0, len(new_batch) - 1)
                    samp = new_batch[rand]
                    new_batch.append(samp)

                if np.random.rand() < 0.2:
                    CocoDataloader.samples.append(samp)
                print("s ", len(new_batch)," r ", rand)

        return torch.utils.data.dataloader.default_collate(new_batch)

train_data_dir = '/home/jansel/Documents/Research/coco_dataset/data/val2017'
train_coco = '/home/jansel/Documents/Research/coco_dataset/data/instances_val2017.json'


# create own Dataset
#my_dataset = CocoDataloader(root=train_data_dir, annotation=train_coco)
#
## Batch size
#train_batch_size = 32
#data_loader = torch.utils.data.DataLoader(my_dataset,
#                                          batch_size=train_batch_size,
#                                          shuffle=True,
#                                          collate_fn=my_dataset.collate_fn
#                                          )
#
## select device (whether GPU or CPU)
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#
## DataLoader is iterable over Dataset
#for img1, img2, af, all1, cat in data_loader:
#    x = 0
##    print(img1.shape, img2.shape, af.shape, all1.shape, cat.shape)
#    print(cat[0])
#    print(cat[0].shape, torch.max(cat[0]))
#
#    img = img1[0].numpy()#tf.ToPILImage()(img1[0] * 255).convert('RGB')
#    img = img.reshape(100, 100, 3)
#    plt.figure(0)
#    plt.imshow(img)
#    img = tf.ToPILImage()(cat[0])
#    plt.figure(1)
#    plt.imshow(img)
#    img = img2[0].numpy()#tf.ToPILImage()(img1[0] * 255).convert('RGB')
#    img = img.reshape(100, 100, 3)
#    plt.figure(2)
#    plt.imshow(img)
#    plt.show()
#    time.sleep(1)
