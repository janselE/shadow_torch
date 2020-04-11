import cv2
import glob
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader

def read_dataset(filename):
    imgs_names = []
    target_names = []
    for filename in glob.glob(filename):
        imgs_names.append(filename)
        s = ""
        s += filename.replace("train_A", "train_C")
        target_names.append(s)

    return imgs_names, target_names

# Create the data class, this is done to load the data into the pytorch model
# this class might be slow because is reading the image at the time is being requested
# in this manner we do not load a lot of the images and run out of memmory
class Data(Dataset):
    # Constructor
    def __init__(self):
        self.imgs_names, self.target_names = read_dataset('../ISTD_Dataset/train/train_A/*.png')
        self.len = len(self.imgs_names)

    # Getter
    def __getitem__(self, index):
        imgs_c = []
        target_c = []

        imgs_c.append(cv2.imread(self.imgs_names[index]))
        target_c.append(cv2.imread(self.target_names[index]))

        imgs_c = np.asarray(imgs_c)
        target_c = np.asarray(target_c)

        imgsr_c = imgs_c.reshape(3, 480, 640).astype('float32') / 255
        targetr_c = target_c.reshape(3, 480, 640).astype('float32') / 255

        # convert the numpy arrays into torch tensors
        t_imgs = torch.tensor(np.asarray(imgsr_c)) # this is to do regression on channels
        t_target = torch.tensor(np.asarray(targetr_c)) # this is to do regression on channels

        self.x = t_imgs.float()
        self.y = t_target.float()

        return self.x, self.y

    # Get items
    def __len__(self):
        return self.len


