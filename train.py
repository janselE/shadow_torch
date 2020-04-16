from network import *
from image_loader import *

dataloader  = DataLoader(dataset=Data(), batch_size=8, shuffle=True) # Create the loader for the model

# Create the models
netD = Discriminator()
netG = Generator()

# Initialize BCELoss function
adversarial_loss = nn.BCELoss()
pixelwise_loss = nn.L1Loss()

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Defining the learning rate, number of epochs and beta for the optimizers
lr = 0.001
beta1 = 0.5
num_epochs = 10

# Setup Adam optimizers for both G and D
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# Training Loop
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        #split the data on the required toruch variables
        original_image, shadow_mask, shadow_free_image = data

        #get the size of the batchsize on that specific iteration(pytorch has an example)
        b_size = original_image.size(0)

        #generate labels
        real  = torch.full((b_size,), real_label) # we can add this to a device by another device parameter
        fake = torch.full((b_size,), fake_label) # we can add this to a device by another device parameter

        #create input
        real_imgs = torch.cat((original_image, shadow_mask), 1)
        print('shape of real {}'.format(real_imgs))

        # Train the generator
        optimizerG.zero_grad()

        fake_imgs = netG(real_imgs)
        fake_mask = torch.zeros(fake_imgs.shape)

        gen_imgs = torch.cat((fake_imgs, fake_mask), 1)
        print('shape of fake {}'.format(gen_imgs))

        lossG_1 = adversarial_loss(netD(gen_imgs).view(-1), real) # maybe .view(-1)
        lossG_2 = pixelwise_loss(gen_imgs, real_imgs) # this would be bad because the mask of generated is black

        lossG = 0.001 * lossG_1 + 0.999 * lossG_2

        lossG.backward()
        optimizerG.step()


        # Train the discriminator on real images
        #zero_grad the discriminator
        optimizerD.zero_grad()

        #run discriminator on real data (reshape the output using .view(-1))
        #calculate the error using the real data (.criterion(output, label))
        lossD_real = adversarial_loss(netD(real_imgs).view(-1), real)
        lossD_fake = adversarial_loss(netD(gen_imgs.detach()).view(-1), fake)
        lossD = 0.5 * (lossD_fake + lossD_real)

        #run the backpropagation (.backward())
        lossD.backward()
        optimizerD.step()
        print(lossG.item(), lossD.item())

    print('Done with the epoch')

