import torch
import torch.nn as nn

# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.feature = nn.Sequential(
                                     nn.Conv2d(4, 64, 3, 1, 1),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(64, 128, 3, 1, 1),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(128, 256, 3, 1, 1),
                                     nn.BatchNorm2d(256),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(256, 512, 3, 1, 1),
                                     nn.BatchNorm2d(512),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(512, 1, 3, 1, 1),
                                     nn.Sigmoid()
                                     )
    def forward(self, input):
        output = self.feature(input)
        return output



# Generator Code
# change this code so that the generator take the image and the mask
class Generator(nn.Module):
    def  __init__(self):
        super(Generator, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(4, 64, 3, 1, 1), nn.LeakyReLU())
        self.conv1 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU())
        self.convt6 = nn.Sequential(nn.ConvTranspose2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.convt7 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.convt8 = nn.Sequential(nn.ConvTranspose2d(1024, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.convt9 = nn.Sequential(nn.ConvTranspose2d(512, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.convt10 = nn.Sequential(nn.ConvTranspose2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.convt11 = nn.Sequential(nn.ConvTranspose2d(128, 1, 3, 1, 1), nn.Tanh())

    def forward(self, input):
        conv0 = self.conv0(input)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv4 = self.conv4(conv4)
        conv4 = self.conv4(conv4)
        conv5 = self.conv5(conv4)
        convt6 = self.convt6(conv5)
        conv6 = torch.cat((conv4, convt6), 1)
        convt7 = self.convt7(conv6)
        conv6 = torch.cat((conv4, convt7), 1)
        convt7 = self.convt7(conv6)
        conv6 = torch.cat((conv4, convt7), 1)
        convt7 = self.convt7(conv6)
        conv7 = torch.cat((conv3, convt7), 1)
        convt8 = self.convt8(conv7)
        conv8 = torch.cat((conv2, convt8), 1)
        convt9 = self.convt9(conv8)
        conv9 = torch.cat((conv1, convt9), 1)
        convt10 = self.convt10(conv9)
        conv10 = torch.cat((conv0, convt10), 1)
        convt11 = self.convt11(conv10)

        return convt11
