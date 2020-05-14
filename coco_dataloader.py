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

class CocoDataloader(torch.utils.data.Dataset):
    samples = []
    classes = []

    def __init__(self, root, annotation, input_sz, use_random_scale=False, use_random_affine=True, transforms=None, classes_path=None):
        self.root = root
        self.transforms = transforms
        self._sorted_coarse_names = [
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

        self._sorted_coarse_name_to_coarse_index = {n: i for i, n in enumerate(self._sorted_coarse_names)}
        self.fine_index_to_coarse_index, self.fine_name_to_coarse_name = self.generate_fine_to_coarse()

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

#        if path not in CocoDataloader.classes and len(CocoDataloader.classes) > 0:
#            return None, None

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

            lab = self.fine_index_to_coarse_index[cat]
            class_name = self._sorted_coarse_names[lab]
            if 'stuff' in class_name:
                print(cat, lab, class_name)

            for n in range(len(seg)):
                # add a condition for taking stuff
                draw.polygon(seg[n], outline=None, fill=lab)
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


    def _find_parent(self, name, d):
        for k, v in d.items():
            if isinstance(v, list):
                  if name in v:
                        yield k
            else:
                assert (isinstance(v, dict))
                for res in self._find_parent(name, v):  # if it returns anything to us
                    yield res

    def getMaps(self, arg1):
        dic = {}
        index = 0
        seg_list = []
        cat_list = []
        curr = 0
        for index in range(len_img):
            img_id = data['images'][index]
            path = img_id['file_name']
            len_ann = len(data['annotations'])
            ann_ids = []
            bad = False

            for i in range(len_):
                if data['annotations'][i]['image_id'] == data['images'][index]['id']:
                    ann_ids.append(i)
            num_objs = len(ann_ids)
            seg_list = []
            cat_list = []
            for i in range(num_objs):
                seg = data['annotations'][ann_ids[i]]['segmentation']
                cat = data['annotations'][ann_ids[i]]['category_id']
                crowd = data['annotations'][ann_ids[i]]['iscrowd']

                seg_list.append(seg)
                cat_list.append(fine_index_to_coarse_index[cat])

                if crowd == 1:
                    bad = True
                    break
                for n in range(len(seg)):
                    lab = fine_index_to_coarse_index[cat]
                    class_name = _sorted_coarse_names[lab]
                    if lab >= 12 or bad:
                        bad = True
                        break
        if bad:
            continue
            dic[str(curr)] = [path, seg_list, cat_list]
            curr += 1

        return dic

    def generate_fine_to_coarse(self):
        fine_index_to_coarse_index = {}
        fine_name_to_coarse_name = {}
    
        with open("data/cocostuff_fine_raw.txt") as f:
            l = [tuple(pair.rstrip().split('\t')) for pair in f]
            l = [(int(ind), name) for ind, name in l]
    
        with open("data/cocostuff_hierarchy.y") as f:
            d = yaml.load(f, Loader=yaml.FullLoader)

        for fine_ind, fine_name in l:
            assert (fine_ind >= 0 and fine_ind < 182)
            parent_name = list(self._find_parent(fine_name, d))
            # print("parent_name of %d %s: %s"% (fine_ind, fine_name, parent_name))
            assert (len(parent_name) == 1)
            parent_name = parent_name[0]
            parent_ind = self._sorted_coarse_name_to_coarse_index[parent_name]
            assert (parent_ind >= 0 and parent_ind < 27)

            fine_index_to_coarse_index[fine_ind] = parent_ind
            fine_name_to_coarse_name[fine_name] = parent_name

        assert (len(fine_index_to_coarse_index) == 182)

        return fine_index_to_coarse_index, fine_name_to_coarse_name

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

        if (len(CocoDataloader.samples) < 128 or len(CocoDataloader.samples) == 0) and len(new_batch) != 0: # this is just a number to limitate the memory usage
            CocoDataloader.samples.append(random.choice(new_batch))

        if len_batch > len(new_batch):
            diff = len_batch - len(new_batch)

            for i in range(diff):
                if len(new_batch) == 0:
                    new_batch.append(random.choice(CocoDataloader.samples))

                else:
                    new_batch.append(random.choice(new_batch))

                if len(CocoDataloader.samples) < 128 or len(CocoDataloader.samples) == 0: # this is just a number to limitate the memory usage
                    CocoDataloader.samples.append(random.choice(new_batch))


        return torch.utils.data.dataloader.default_collate(new_batch)
