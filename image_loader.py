import cv2
import glob
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader

# take as parameter the filename that we already know
# we are just changing the part of the name that is different
# this method has to be changed for different datasets
def read_dataset(filename):
    imgs_names = []
    target_names = []
    mask_names = []
    for filename in glob.glob(filename):
        imgs_names.append(filename)
        s = ""
        s += filename.replace("train_A", "train_B")
        mask_names.append(s)
        mask_names.append(s)
        s = ""
        s += filename.replace("train_A", "train_C")
        target_names.append(s)

    return imgs_names, mask_names, target_names

# Create the data class, this is done to load the data into the pytorch model
# this class might be slow because is reading the image at the time is being requested
# in this manner we do not load a lot of the images and run out of memmory
class Data(Dataset):
    # Constructor
    def __init__(self):
        self.imgs_names, self.mask_names, self.target_names = read_dataset('../../../ISTD_Dataset/train/train_A/*.png') # known name
        self.len = len(self.imgs_names) # here we take the size of all the names that we can use

    # Getter
    def __getitem__(self, index):

        # in this method we load the images when requested
        imgs_c = cv2.imread(self.imgs_names[index])
        mask_c = cv2.imread(self.mask_names[index], 0)
        target_c = cv2.imread(self.target_names[index])

        imgs_c = np.asarray(imgs_c)
        mask_c = np.asarray(mask_c)
        target_c = np.asarray(target_c)

        imgsr_c = imgs_c.reshape(3, 480, 640).astype('float32') / 255
        maskr_c = mask_c.reshape(1, 480, 640).astype('float32') / 255
        targetr_c = target_c.reshape(3, 480, 640).astype('float32') / 255

        # convert the numpy arrays into torch tensors
        t_imgs = torch.tensor(np.asarray(imgsr_c)) # this is to do regression on channels
        t_mask = torch.tensor(np.asarray(maskr_c)) # this is to do regression on channels
        t_target = torch.tensor(np.asarray(targetr_c)) # this is to do regression on channels

        self.x = t_imgs.float()
        self.y = t_mask.float()
        self.z = t_target.float()

        return self.x, self.y, self.z

    # Get items
    def __len__(self):
        return self.len


