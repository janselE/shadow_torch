import os
import torch
import torch.utils.data
import torchvision
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import numpy as np

class CocoDataloader(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))
        (w, h) = img.size

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        mask = Image.new('L', (w, h), 0)
        for i in range(num_objs):
            seg = coco_annotation[i]['segmentation']
            cat = coco_annotation[i]['category_id']
            for n in range(len(seg)):
                ImageDraw.Draw(mask).polygon(seg[n], fill=cat)

        img = np.asarray(img)
        mask = np.asarray(mask)
        print(img.shape, mask.shape)

        img = img.astype(np.float32) / 255
        mask = mask.astype(np.float32) / 255

        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

#        if self.transforms is not None:
#            img = self.transforms(img)

        return img, mask 

    def __len__(self):
        return len(self.ids)


train_data_dir = '/home/jansel/Documents/Research/coco_dataset/data/val2017'
train_coco = '/home/jansel/Documents/Research/coco_dataset/data/instances_val2017.json'

# create own Dataset
my_dataset = CocoDataloader(root=train_data_dir,
                          annotation=train_coco,
                          )

# Batch size
train_batch_size = 1
data_loader = torch.utils.data.DataLoader(my_dataset,
                                          batch_size=train_batch_size,
                                          shuffle=True,
                                          num_workers=4,
                                          )

# select device (whether GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# DataLoader is iterable over Dataset
for imgs, mask in data_loader:
    print(imgs.shape, mask.shape)

