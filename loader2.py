import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader

# scripts
from image_loader import Data
from net10a import SegmentationNet10a
from IIC_Losses import IID_segmentation_loss, IID_segmentation_loss_uncollapsed


h, w, in_channels = 240, 240, 3

# Create the models
net = SegmentationNet10a()  # One head. We cannot use overclustering (with second head) since we don't have additional class labels

# Defining the learning rate, number of epochs and beta for the optimizers
lr = 0.001
beta1 = 0.5
num_epochs = 10

# Initialize IIC objective function
loss_fn = IID_segmentation_loss

# Setup Adam optimizers for both
optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.1))

# Lists to keep track of progress
img_list = []
# G_losses = []  # maybe make sure just shadow mask prediction is working first before we test GAN
# D_losses = []
iters = 0
# heads = ["A", "B"]
# epoch_loss_head_A = []
epoch_loss = []
# epoch_loss_head_B = []
# lamb_A = 1.0
lamb = 1.0
# lamb_B = 1.0
batch_sz = 8
num_sub_heads = 2  # what is a subhead?
half_T_side_dense = 0
half_T_side_sparse_min = 0
half_T_side_sparse_max = 0

transform = transforms.Compose([transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()])  # the transform applied to create the second image
dataloader = DataLoader(dataset=Data(h, w,"A", transform),
        batch_size=batch_sz, shuffle=True)  # Loads image and transformed image, both from shadow dataset

# dataloaders_list = [dataloader_A, dataloader_B]

## For each epoch
#for epoch in range(num_epochs):
#    # For each batch in the dataloader
#    for i, data in enumerate(dataloader, 0):
#        original_image, transformed = data

for epoch in range(0, num_epochs):
    print("Starting epoch: %d " % (epoch))

#    if e_i in config.lr_schedule:
#        optimiser = update_lr(optimiser, lr_mult=config.lr_mult)

    # for head_i in range(2):
    #     head = heads[head_i]
    # if head == "A":
    # dataloaders = dataloaders_list
    # epoch_loss = epoch_loss_head_A
    #epoch_loss_no_lamb = config.epoch_loss_no_lamb_head_A
    # lamb = lamb_A  # what is lamb? stands for lambda? is hyperparameter?

    # elif head == "B":
    #     dataloaders = dataloaders_list
    #     epoch_loss = epoch_loss_head_B
    #     #epoch_loss_no_lamb = config.epoch_loss_no_lamb_head_B
    #     lamb = lamb_B

    # iterators = (d for d in dataloaders)  # why two dataloaders? For image and transformed image?
    batch_num = 0
    avg_loss = 0.  # over heads and head_epochs (and sub_heads)
    avg_loss_no_lamb = 0.
    avg_loss_count = 0

    for tup in zip(*dataloader):
        print(type(tup))
        net.zero_grad()

        pre_channels = in_channels

        all_img1 = torch.zeros(batch_sz, pre_channels, h, w).to(torch.float32) #cuda
        all_img2 = torch.zeros(batch_sz, pre_channels, h, w).to(torch.float32) # cuda
        all_affine2_to_1 = torch.zeros(batch_sz, 2, 3).to(torch.float32) # cuda
        all_mask_img1 = torch.zeros(batch_sz, h, w).to(torch.float32) # cuda

        curr_batch_sz = tup[0][0].shape[0]  # batch size is determined by dataloader when constructed
        # for d_i in range(2): # verify this, I think it does matter the amout of dataloaders
        img1, img2 = tup[0]  # so one dataloader provides the 2 images we want to compare? Then why are there 2 dataloaders?
        print(img1.shape, img2.shape)
        #img1, img2, affine2_to_1, mask_img1 = tup[d_i]
        assert (img1.shape[0] == curr_batch_sz)

        # actual_batch_start = d_i * batch_sz  # why do we need to keep track of this? Wouldn't need it with only one dataloader?
        # actual_batch_end = actual_batch_start + batch_sz

        # all_img1 = img1  # why? why initialize all_img1 if doing this?
        # all_img2 = img2
        #all_img1[actual_batch_start:actual_batch_end, :, :, :] = img1
        #all_img2[actual_batch_start:actual_batch_end, :, :, :] = img2
        #all_affine2_to_1[actual_batch_start:actual_batch_end, :, :] = affine2_to_1
        #all_mask_img1[actual_batch_start:actual_batch_end, :, :] = mask_img1


        # curr_total_batch_sz = batch_sz * 2 #num_dataloaders  # times 2
        # all_img1 = all_img1[:curr_total_batch_sz, :, :, :]
        # all_img2 = all_img2[:curr_total_batch_sz, :, :, :]
        #all_affine2_to_1 = all_affine2_to_1[:curr_total_batch_sz, :, :]
        #all_mask_img1 = all_mask_img1[:curr_total_batch_sz, :, :]

        # erased the no sobel if statement

        x1_outs = net(img1)
        x2_outs = net(img2)

        avg_loss_batch = None  # avg over the heads
        avg_loss_no_lamb_batch = None

        for i in range(num_sub_heads):
            print("This is the shape {}".format(x1_outs.shape))
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
        print("e {} b {} al {} aln {}".format(epoch, batch_num, avg_loss_batch.item(),
                                              avg_loss_no_lamb_batch.item()))

        if not np.isfinite(avg_loss_batch.item()):
            print("Loss is not finite... %s:" % str(avg_loss_batch))
            exit(1)

        avg_loss += avg_loss_batch.item()
        avg_loss_no_lamb += avg_loss_no_lamb_batch.item()
        avg_loss_count += 1

        avg_loss_batch.backward()
        optimizer.step()

        torch.cuda.empty_cache()

        batch_num += 1
        if batch_num == 2 and config.test_code:
            break

        # end of heads loop

        avg_loss = float(avg_loss / avg_loss_count)
        avg_loss_no_lamb = float(avg_loss_no_lamb / avg_loss_count)

        epoch_loss.append(avg_loss)
        epoch_loss_no_lamb.append(avg_loss_no_lamb)  # what is this?

