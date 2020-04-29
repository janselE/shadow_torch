
from image_loader import *

h, w, in_channels = 240, 240, 3
batch_sz = 32

dataloader = DataLoader(dataset=FullDataset(h, w, use_random_scale=False, use_random_affine=True),
                        batch_size=batch_sz, shuffle=True, drop_last=True)  # shuffle is to pick random images and drop last is to drop the last batch so the size does not changes


for idx, data in enumerate(dataloader):
    img1, img2, cat, afine, mask = data
    print(img1.shape, img2.shape, cat.shape, afine.shape, mask.shape)

