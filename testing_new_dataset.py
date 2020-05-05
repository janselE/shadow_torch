
from image_loader import *
from coco_dataloader import CocoDataloader

h, w, in_channels = 240, 240, 3
batch_sz = 32

train_data_dir = '/home/jansel/Documents/Research/coco_dataset/data/val2017'
train_coco = '/home/jansel/Documents/Research/coco_dataset/data/instances_val2017.json'
dataloader = DataLoader(dataset=CocoDataloader(root=train_data_dir, annotation=train_coco, input_sz=h),
                        batch_size=batch_sz, shuffle=True, collate_fn=CocoDataloader.collate_fn, drop_last=True)
#dataloader = DataLoader(dataset=ShadowAndMaskDataset(h, w, use_random_scale=False, use_random_affine=True),
#                        batch_size=batch_sz, shuffle=True, drop_last=True)  # shuffle is to pick random images and drop last is to drop the last batch so the size does not changes


for idx, data in enumerate(dataloader):
    img1, img2, cat, afine, mask = data
    print(img1.shape, img2.shape, cat.shape, afine.shape, mask.shape)

