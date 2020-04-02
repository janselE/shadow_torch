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

imgs_g, target_g, imgs_c, target_c = reader('../ISTD_Dataset/train/train_A/*.png')

#process the images
imgsr = imgs_g.reshape(-1, 480, 640, 1).astype('float32') / 255
targetr = target_g.reshape(-1, 480, 640, 1).astype('float32') / 255

imgsr_c = imgs_c.reshape(-1, 480, 640, 3).astype('float32') / 255
targetr_c = target_c.reshape(-1, 480, 640, 3).astype('float32') / 255

print(imgsr.shape,  targetr.shape)
print(imgsr_c.shape,  targetr_c.shape)



import matplotlib.pyplot as plt

# This output the images that are being used and generated
# First column is for the training image (original)
# Third column is the target values (gt)
for i in range(20):
    fig, ax = plt.subplots(1, 2, sharex='col', sharey='row')
    for j in range(3):
        if j == 0:
            ax[j].imshow(imgs_c[i])
        elif j == 1:
            ax[j].imshow(target_c[i])



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


class SimpleNet(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(SimpleNet, self).__init__()

        self.i = nn.Linear(inputSize, 32)
        self.relu1 = nn.ReLU()

        self.h2 = nn.Linear(32, 64)
        self.relu2 = nn.ReLU()

        self.h3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()

        self.o = nn.Linear(32, outputSize)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        output1 = self.i(x)
        output1 = self.relu1(output1)

        output2 = self.h2(output1)
        output2 = self.relu2(output2)

        output3 = self.h3(output2)
        output3 = self.relu3(output3)

        output4 = output1 + output3 # skip connection

        output = self.o(output4)

        return output


trainloader = DataLoader(dataset=Data(), batch_size=32) # Create the loader for the model 
model = SimpleNet(1, 1) # Initialize the model


learningRate = 1
epochs = 10

criterion = torch.nn.MSELoss() # Loss function
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate) # Gradient


# Train the model
def train_model(epochs):
    for epoch in range(epochs):
        for x, y in trainloader:
            yhat = model(x)
            print("{} {}".format(yhat.shape, "this is the shape"))

            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch {}, loss {}'.format(epoch, loss.item()))
        if epoch % 10 ==0:
            save_models(epoch)


train_model(10)

torch.cuda.is_available()
