from network import *
from image_loader import *

dataloader  = DataLoader(dataset=Data(), batch_size=16, shuffle=True) # Create the loader for the model

# Create the models
netD = Discriminator()
netG = Generator()

# Initialize BCELoss function
criterionD = nn.BCELoss()
criterionG_1 = nn.BCELoss()
criterionG_2 = nn.L1Loss()

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

        #zero_grad the discriminator
        netD.zero_grad()

        #get the size of the batchsize on that specific iteration(pytorch has an example)
        b_size = original_image.size(0)

        #generate real labels
        label = torch.full((b_size,), real_label) # we can add this to a divice by another device parameter

        #create input
        input_tensor = torch.cat((original_image, shadow_mask), 1)
        print("input shape {}".format(input_tensor.shape))

        # Train the discriminator on real images
        #run discriminator on real data (reshape the output using .view(-1))
        output = netD(input_tensor).view(-1)
        print(output.shape)

        #calculate the error using the real data (.criterion(output, label))
        lossD_real = criterionD(output, label)

        #run the backpropagation (.backward())
        lossD_real.backward()

        # Train the discriminator on fake images
        #generate fake images
        fake = netG(input_tensor)
        fake_mask = torch.zeros(fake.shape)
        gen_input_tensor = torch.cat((fake.detach(), fake_mask), 1)
        gen_input_tensor = gen_input_tensor

        #generate fake labels (.fill(fake_labels))
        label = label.fill_(fake_labels)
        #run the discriminator on fake images (detach the output of the generator) also, reshape (.view(-1))
        output = netD(gen_input_tensor).view(-1)
        #calculate the error using the fake data (.criterion(output, label))
        lossD_fake = criterionD(output, label)
        #run the backpropagation (.backward())
        lossD_fake = backward()
        #add both errors of the discriminator
        loss = lossD_fake + lossD_real
        #update the discriminator (optimizer.step())
        optimizerD.step()

        # Train the generator
        #zero_grad the generator
        netG.zero_grad()
        #generate real labels for the generator (they are real for the generator)
        label = label.fill_(real_label)
        #use the previouse output of the discriminator to calculate the cost of the generator (criterion(output, label))
        lossG_1 = criterionG_1(output, label)
        lossG_1.backward()

        lossG_2 = criterionG_2(fake, input_tensor)
        lossG_2.backward()

        lossG = lossG_1 + lossG_2
        #run the backpropagation (.backward())
        #update the generator (optimizer.step())
        optimizerG.step()

        break
