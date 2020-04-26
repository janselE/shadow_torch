import torchvision
import torchvision.transforms as transforms
import torch

# scripts
from image_loader import *
from net10a_twohead import *
from IIC_Losses import *


h, w, in_channels = 240, 240, 3

# Create the models
net = SegmentationNet10a()

# Defining the learning rate, number of epochs and beta for the optimizers
lr = 0.001
beta1 = 0.5
num_epochs = 10

# Initialize IIC objective function
loss_fn = IID_segmentation_loss

# Setup Adam optimizers for both
optimiser = torch.optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.1))

# Lists to keep track of progress
img_list = []
G_losses = []  # maybe make sure just shadow mask prediction is working first before we test GAN
D_losses = []
iters = 0
epoch_loss_head_A = []
epoch_loss_head_B = []
lamb = 1.0
batch_sz = 8
num_sub_heads = 2
half_T_side_dense = 0
half_T_side_sparse_min = 0
half_T_side_sparse_max = 0

transform = transforms.Compose([transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()])
# Creates a dataloader for the model
dataloader = DataLoader(dataset=Data(h, w, transform),  # what data does this load? we just want to load images with shadows, right?
        batch_size=batch_sz, shuffle=True, drop_last=True)  # shuffle is to pick random images and drop last is to drop the last batch so the size does not changes
dataloaders = [dataloader, dataloader]


## For each epoch
#for epoch in range(num_epochs):
#    # For each batch in the dataloader
#    for i, data in enumerate(dataloader, 0):
#        img1, img2, aff, mask = data
#        print(img1.shape, img2.shape, aff.shape, mask.shape)

for e_i in range(0, num_epochs):
    print("Starting e_i: %d " % (e_i))

#    if e_i in config.lr_schedule:
#        optimiser = update_lr(optimiser, lr_mult=config.lr_mult)


    iterators = (d for d in dataloaders)  # why two dataloaders? For image and transformed image?
    b_i = 0
    avg_loss = 0.  # over heads and head_epochs (and sub_heads)
    avg_loss_no_lamb = 0.
    avg_loss_count = 0

    for tup in zip(*iterators):
        net.zero_grad()

        pre_channels = in_channels
        num_dataloaders = len(dataloaders)

        all_img1 = torch.zeros(batch_sz * num_dataloaders, pre_channels, h, w).to(torch.float32) #cuda
        all_img2 = torch.zeros(batch_sz *num_dataloaders, pre_channels, h, w).to(torch.float32) # cuda
        all_affine2_to_1 = torch.zeros(batch_sz * num_dataloaders, 2, 3).to(torch.float32) # cuda
        all_mask_img1 = torch.zeros(batch_sz * num_dataloaders, h, w).to(torch.float32) # cuda

        curr_batch_sz = tup[0][0].shape[0]
        for d_i in range(num_dataloaders): # verify this, I think it does matter the amout of dataloaders
            img1, img2, affine2_to_1, mask_img1 = tup[d_i]
            assert (img1.shape[0] == curr_batch_sz)

            actual_batch_start = d_i * batch_sz  # why do we need to keep track of this? Wouldn't need it with only one dataloader?
            actual_batch_end = actual_batch_start + batch_sz

            all_img1[actual_batch_start:actual_batch_end, :, :, :] = img1
            all_img2[actual_batch_start:actual_batch_end, :, :, :] = img2
            all_affine2_to_1[actual_batch_start:actual_batch_end, :, :] = affine2_to_1
            all_mask_img1[actual_batch_start:actual_batch_end, :, :] = mask_img1

        curr_total_batch_sz = batch_sz * num_dataloaders  # times 2
        all_img1 = all_img1[:curr_total_batch_sz, :, :, :]
        all_img2 = all_img2[:curr_total_batch_sz, :, :, :]
        all_affine2_to_1 = all_affine2_to_1[:curr_total_batch_sz, :, :]
        all_mask_img1 = all_mask_img1[:curr_total_batch_sz, :, :]

        # erased the no sobel if statement

        x1_outs = net(all_img1)
        x2_outs = net(all_img2)

        avg_loss_batch = None  # avg over the heads
        avg_loss_no_lamb_batch = None

        for i in range(num_sub_heads):
            loss, loss_no_lamb = loss_fn(x1_outs[i], x2_outs[i],
                    all_affine2_to_1=all_affine2_to_1,
                    all_mask_img1=all_mask_img1, lamb=lamb,
                    half_T_side_dense=half_T_side_dense,
                    half_T_side_sparse_min=half_T_side_sparse_min,
                    half_T_side_sparse_max=half_T_side_sparse_max)

            if avg_loss_batch is None:
                avg_loss_batch = loss
                avg_loss_no_lamb_batch = loss_no_lamb
            else:
                avg_loss_batch += loss
                avg_loss_no_lamb_batch += loss_no_lamb

        avg_loss_batch /= num_sub_heads
        avg_loss_no_lamb_batch /= num_sub_heads

        # check the print statement that was here
        print("e {} b {} al {} aln {}".format(e_i, b_i, avg_loss_batch.item(),
            avg_loss_no_lamb_batch.item()))

        if not np.isfinite(avg_loss_batch.item()):
            print("Loss is not finite... %s:" % str(avg_loss_batch))
            exit(1)

        avg_loss += avg_loss_batch.item()
        avg_loss_no_lamb += avg_loss_no_lamb_batch.item()
        avg_loss_count += 1

        avg_loss_batch.backward()
        optimiser.step()

        torch.cuda.empty_cache()

        b_i += 1
        if b_i == 2 and config.test_code:
            break

    avg_loss = float(avg_loss / avg_loss_count)
    avg_loss_no_lamb = float(avg_loss_no_lamb / avg_loss_count)
