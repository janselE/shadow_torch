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
    def __init__(self, device):
        self.x = t_imgs.float().to(device)
        self.y = t_target.float().to(device)
        self.len = self.x.shape[0]

    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get items
    def __len__(self):
        return len

print(t_imgs.size(), t_target.size())


# This method is to save the model after a threshold
def save_models(epoch):
    torch.save(model.state_dict(), "reg_model{}.model".format(epoch))
    print("Chekcpoint saved")


class UNet(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels),
                torch.nn.ReLU(), torch.nn.BatchNorm2d(out_channels), torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels),
                torch.nn.ReLU(), torch.nn.BatchNorm2d(out_channels)
                )

        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel),
                torch.nn.ReLU(), torch.nn.BatchNorm2d(mid_channel), torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
                torch.nn.ReLU(), torch.nn.BatchNorm2d(mid_channel),
                torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                )
        return  block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel),
                torch.nn.ReLU(), torch.nn.BatchNorm2d(mid_channel), torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
                torch.nn.ReLU(), torch.nn.BatchNorm2d(mid_channel), torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
                torch.nn.ReLU(), torch.nn.BatchNorm2d(out_channels))
        return  block

    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        #Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=64)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(64, 128)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(128, 256)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = torch.nn.Sequential(torch.nn.Conv2d(kernel_size=3, in_channels=256, out_channels=512),
                torch.nn.ReLU(), torch.nn.BatchNorm2d(512),
                torch.nn.Conv2d(kernel_size=3, in_channels=512, out_channels=512),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(512),
                torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
                )

        # Decode
        self.conv_decode3 = self.expansive_block(512, 256, 128)
        self.conv_decode2 = self.expansive_block(256, 128, 64)
        self.final_layer = self.final_block(128, 64, out_channel)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)

        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)

        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=True)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)
        final_layer = self.final_layer(decode_block1)

        return  final_layer




#class ConvNet(nn.Module):
#    def __init__(:
#        super(ConvNet, .__init__()
#        layer1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
#        relu1 = nn.ReLU()
#
#        layer2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
#        relu2 = nn.ReLU()
#
#        layer3 = nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2)
#        sig = nn.Sigmoid()
#
#    def forward( x):
#        out = layer1(x)
#        out = relu1(out)
#
#        out = layer2(out)
#        out = relu2(out)
#
#        out = layer3(out)
#        out = sig(out)
#
#        return out

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on GPU")
else:
    device = torch.device("cpu")
    print("running on CPU")

model = UNet(1, 1).to(device) # Initialize the model
trainloader = DataLoader(dataset=Data(device), batch_size=32) # Create the loader for the model 

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

print(torch.cuda.is_available())
