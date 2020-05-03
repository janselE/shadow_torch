import os
import torch
import torch.utils.data
import torchvision
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import numpy as np
import random


class CocoDataloader(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def getImg(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path))

        return img, coco_annotation

    def __getitem__(self, index):
        (w, h) = (100, 100)
        img, coco_annotation = self.getImg(index)

        # number of objects in the image
        num_objs = len(coco_annotation)

        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        for i in range(num_objs):
            seg = coco_annotation[i]['segmentation']
            cat = coco_annotation[i]['category_id']
            crowd = coco_annotation[i]['iscrowd']
            if crowd == 1:
                return None
            for n in range(len(seg)):
                draw.polygon(seg[n], outline=None, fill=cat)
        del draw

        img = img.resize((w,h))
        mask = mask.resize((w,h))

        img = np.asarray(img)
        mask = np.asarray(mask)
#        print(img.shape, mask.shape)

        img = img.astype(np.float32) / 255
        mask = mask.astype(np.float32) / 255

        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

#        if self.transforms is not None:
#            img = self.transforms(img)

        return [img, mask]

    def __len__(self):
        return len(self.ids)

train_data_dir = '/home/jansel/Documents/Research/coco_dataset/data/val2017'
train_coco = '/home/jansel/Documents/Research/coco_dataset/data/instances_val2017.json'

def collate_fn(batch):
    len_batch = len(batch) # original batch length
    batch = list(filter (lambda x:x is not None, batch)) # filter out all the Nones
    if len_batch > len(batch): # if there are samples missing just use existing members, doesn't work if you reject every sample in a batch
        diff = len_batch - len(batch)
        for i in range(diff):
            rand = random.randint(0, diff)
            samp = batch[rand]
            print(samp[0].shape, samp[1].shape)
            batch.append(samp)
    return torch.utils.data.dataloader.default_collate(batch)

# create own Dataset
my_dataset = CocoDataloader(root=train_data_dir,
                          annotation=train_coco,
                          )

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
for imgs, mask in data_loader:
    x =0

