import cv2
import glob
import numpy as np
from torch.autograd import Variable
import torch.nn as nn

def read_dataset(filename):
    imgs_names = []
    target_names = []
    for filename in glob.glob(filename):
        imgs_names.append(filename)
        s = ""
        s += filename.replace("train_A", "train_C")
        target_names.append(s)

    return imgs_names, target_names


def reader(filename):
    imgs_g = []
    target_g = []

    imgs_c = []
    target_c = []


    imgs_names, target_names = read_dataset(filename)
    amt = 20

    # Loop that read in the images and the target images
    for i in range(0, amt):
        imgs_g.append(cv2.imread(imgs_names[i], 0))
        target_g.append(cv2.imread(target_names[i], 0))

        imgs_c.append(cv2.imread(imgs_names[i]))
        target_c.append(cv2.imread(target_names[i]))

    imgs_g = np.asarray(imgs_g)
    target_g = np.asarray(target_g)

    imgs_c = np.asarray(imgs_c)
    target_c = np.asarray(target_c)


    return imgs_g, target_g, imgs_c, target_c

imgs_g, target_g, imgs_c, target_c = reader('../../../ISTD_Dataset/train/train_A/*.png')

#process the images
imgsr = imgs_g.reshape(-1, 1, 480, 640).astype('float32') / 255
targetr = target_g.reshape(-1, 1, 480, 640).astype('float32') / 255

imgsr_c = imgs_c.reshape(-1, 3, 480, 640).astype('float32') / 255
targetr_c = target_c.reshape(-1, 3, 480, 640).astype('float32') / 255

print(imgsr.shape,  targetr.shape)
print(imgsr_c.shape,  targetr_c.shape)

targetr.shape

import torch
# converting the lists into numpy arrays
# n_imgs = np.asarray(imgs)/255
# n_target = np.asarray(target)/255

# convert the numpy arrays into torch tensors
t_imgs = torch.tensor(np.asarray(imgsr)) # this is to do regression on channels
t_target = torch.tensor(np.asarray(targetr)) # this is to do regression on channels

from torch.utils.data import Dataset, DataLoader

# Create the data class, this is done to load the data into the pytorch model
class Data(Dataset):
    # Constructor
    def __init__(self):
        self.x = t_imgs.float()
        self.y = t_target.float()
        self.len = self.x.shape[0]

    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get items
    def __len__(self):
        return self.len

print(t_imgs.size(), t_target.size())


# This method is to save the model after a threshold
def save_models(epoch):
    torch.save(model.state_dict(), "reg_model{}.model".format(epoch))
    print("Chekcpoint saved")

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.layeri = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2).cuda(0)
        self.layer0 = nn.Conv2d(16, 1, (3,3), 1).cuda(0)
        #self.layer1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2).cuda(0)
        #self.relu1 = nn.ReLU().cuda(0)

        #self.layer2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2).cuda(0)
        #self.relu2 = nn.ReLU().cuda(0)

        #self.layer3 = nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2).cuda(1)
        self.sig = nn.Sigmoid().cuda(1)

    def forward(self, x):
        out = self.layeri(x)
        print(out.shape)
        out = self.layer0(out)
        out = out.to(1)
        #out = self.relu1(out)

        #out = self.layer2(out)
        #out = self.relu2(out)

        #out = out.to(1)

        #out = self.layer3(out)
        out = self.sig(out)
        print(out.shape)

        return out

#if torch.cuda.is_available():
#    device = torch.device("cuda:0")
#    print("running on GPU")
#else:
#    device = torch.device("cpu")
#    print("running on CPU")

model = UNet() # Initialize the model
trainloader = DataLoader(dataset=Data(), batch_size=16) # Create the loader for the model 

print(model)

learningRate = 0.001
epochs = 10

#criterion = torch.nn.MSELoss() # Loss function
#optimizer = torch.optim.SGD(model.parameters(), lr=learningRate) # Gradient
criterion = torch.nn.BCELoss() # Loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate) # Gradient

# Train the model
def train_model(epochs):
    for epoch in range(epochs):
        for x, y in trainloader:
            x = x.to(0)
            y = y.to(1)

            optimizer.zero_grad() # Clears out the gradients
            pred = model(x) # Forward pass
            loss = criterion(pred, y) # Compute loss function
            loss.backward() # backward pass
            optimizer.step() # Optimizer step

        print('epoch {}, loss {}'.format(epoch, loss.item()))
        if epoch % 10 ==0:
            save_models(epoch)

        print(x.shape, y.shape)

        del x
        del y
        torch.cuda.empty_cache()


train_model(10)

print("Starting prediction")
with torch.no_grad():
    predicted = model(t_imgs.float().cuda(0)).cpu().data.numpy()
#    org = t_imgs.numpy()
#    res = t_target.numpy()

print("Done predicting, staring to write on files")

predicted = predicted.reshape(-1, 480, 640)
#org = org.reshape(-1, 480, 640)
#res = res.reshape(-1, 480, 640)

for i in range(0, len(predicted)):
    name = 'prediction/img' + str(i) + '.png'
    cv2.imwrite(name, predicted[i] * 255)

print("Done writing")
