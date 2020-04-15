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

def reader(filename):
    imgs_c = []
    mask = []
    target_c = []


    imgs_names, mask_names, target_names = read_dataset(filename)
    amt = 20

    # Loop that read in the images and the target images
    for i in range(0, amt):
        imgs_c.append(cv2.imread(imgs_names[i]))
        mask.append(cv2.imread(mask_names[i]))
        target_c.append(cv2.imread(target_names[i]))

    imgs_c = np.asarray(imgs_c)
    mask = np.asarray(mask)
    target_c = np.asarray(target_c)


    return imgs_c, mask, target_c

# Create the data class, this is done to load the data into the pytorch model
# this class might be slow because is reading the image at the time is being requested
# in this manner we do not load a lot of the images and run out of memmory
class Data(Dataset):
    # Constructor
    def __init__(self):
        self.imgs, self.mask, self.target = reader('../ISTD_Dataset/train/train_A/*.png') # known name, this is for local

        self.imgsr_c = self.imgs.reshape(3, 480, 640).astype('float32') / 255
        self.maskr_c = self.mask.reshape(1, 480, 640).astype('float32') / 255
        self.targetr_c = self.target.reshape(3, 480, 640).astype('float32') / 255

        # convert the numpy arrays into torch tensors
        self.t_imgs = torch.tensor(np.asarray(self.imgsr_c), requires_grad=False) # this is to do regression on channels
        self.t_mask = torch.tensor(np.asarray(self.maskr_c), requires_grad=False) # this is to do regression on channels
        self.t_target = torch.tensor(np.asarray(self.targetr_c), requires_grad=False) # this is to do regression on channels

        self.x = self.t_imgs.float()
        self.y = self.t_mask.float()
        self.z = self.t_target.float()

        #self.imgs_names, self.mask_names, self.target_names = read_dataset('../../../ISTD_Dataset/train/train_A/*.png') # known name, this is for the server
        self.len = 20
        #self.len = len(self.imgs_names) # here we take the size of all the names that we can use

    # Getter
    def __getitem__(self, index):

        return self.x[index], self.y[index], self.z[index]

    # Get items
    def __len__(self):
        return self.len


