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
    def __init__(self, root, annotation, transforms=None):
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

        self.input_sz = 100

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

    def display(self, im):
        im.show()

    def __getitem__(self, index):
        img, mask = self.getImgAndMask(index, self.input_sz, self.input_sz)

        if img is None:
            return None

    #    t1=Thread(target=self.display,args=(img,))
    #    t1.start()
    #    t2=Thread(target=self.display,args=(mask,))
    #    t2.start()

    #    time.sleep(100000)

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

        if False:#self.use_random_affine:
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

train_data_dir = '/home/jansel/Documents/Research/coco_dataset/data/val2017'
train_coco = '/home/jansel/Documents/Research/coco_dataset/data/instances_val2017.json'

def collate_fn(batch):
    len_batch = len(batch)
    batch = list(filter (lambda x:x is not None, batch))

    if len_batch > len(batch):
        diff = len_batch - len(batch)
        for i in range(diff):
            rand = random.randint(0, diff)
            samp = batch[rand]
            batch.append(samp)

    return torch.utils.data.dataloader.default_collate(batch)

# create own Dataset
my_dataset = CocoDataloader(root=train_data_dir, annotation=train_coco)

# Batch size
train_batch_size = 32
data_loader = torch.utils.data.DataLoader(my_dataset,
                                          batch_size=train_batch_size,
                                          shuffle=True,
                                          collate_fn=collate_fn
                                          )

# select device (whether GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# DataLoader is iterable over Dataset
for img1, img2, af, all1, cat in data_loader:
    print(img1.shape, img2.shape, af.shape, all1.shape, cat.shape)
#    img = tf.ToPILImage()(img1[0] * 255).convert('RGB')
#    img.show()
#    plt.figure(0)
#    plt.imshow(img)
#    plt.show()

