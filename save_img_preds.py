import torch
import os
from net10a_twohead import SegmentationNet10a
from models_for_gan import Generator_inpaint
import torchvision.transforms as transforms
from PIL import Image

IMAGES_PATH = ''
SAVED_MODEL_PATH = ''
SAVE_IMGS_PATH = ''
catx = 0

# load architechtures
IIC = SegmentationNet10a(num_sub_heads, NUM_CLASSES).cuda()
Gen = Generator_inpaint().cuda()

# load trained weights from state dicts
IIC.load_state_dict(SAVED_MODEL_PATH['IIC_state_dict'])
Gen.load_state_dict(SAVED_MODEL_PATH['Gen_state_dict'])


for img_path in os.listdir(IMAGES_PATH):
    img = Image.open(img_path)
    resized_img = transforms.Resize((256, 512))
    img_t = transforms.ToTensor()(resized_img)
    x1_outs = IIC(img_t)  # get segmentation map prediction from each subhead
    seg = x1_outs[0]  # just use prediction from first subhead (we usually only use 1 subhead anyways)
    img_no_catx = remove_catx(img, seg, catx)  # zeroes out pixels predicted as catx
    img_filled = Gen(img_no_catx)  # fills in zeroed out pixels with category other than catx, hopefully realistic
    img_filled = transforms.ToPILImage()(img_filled)
    filename = IMAGES_PATH + '/' + img_path + "catx_removed"
    img_filled.save(filename)

