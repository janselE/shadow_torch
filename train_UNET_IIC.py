import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
from datetime import datetime
import os

# scripts
from image_loader import ShadowDataset, ShadowAndMaskDataset
from coco_dataloader_with_mask import CocoDataloader

#from net10a import SegmentationNet10a
from IIC_Losses import IID_segmentation_loss, IID_segmentation_loss_uncollapsed
from IIC_Network import net

from torch.utils.tensorboard import SummaryWriter

from eval import eval_acc
from color_segmentation import Color_Mask


h, w, in_channels = 240, 240, 3
input_sz = h

# Lists to keep track of progress
ave_losses = []

# keep track of folders and saved files when model is run multiple times
time_begin = str(datetime.now()).replace(' ', '-')
os.mkdir("img_visual_checks/"+time_begin)
os.mkdir("loss_csvs/"+time_begin)

board = "boards/" + time_begin
os.mkdir(board)

writer = SummaryWriter(board)

lamb = 1.0  # will make loss equal to loss_no_lamb
batch_sz = 2
num_sub_heads = 1
half_T_side_dense = 0
half_T_side_sparse_min = 0
half_T_side_sparse_max = 0

# Defining the learning rate, number of epochs and beta for the optimizers
lr = 0.001
beta1 = 0.5
num_epochs = 500
decay = 0.1
n_epochs_stop = 10
epochs_no_improve = 0
min_val_loss = np.Inf
use_supervised = False# set to epoch < num or similar condition?
total_train = 0
correct_train = 0

# Create the models
#net = SegmentationNet10a(num_sub_heads, 12)
net = net()
net.cuda()

# Initialize IIC objective function
#loss_fn = IID_segmentation_loss
loss_fn = IID_segmentation_loss_uncollapsed
#criterion_ssm = torch.nn.NLLLoss()  # supervised shadow mask loss function

# Setup Adam optimizers for both
optimiser = torch.optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.1))

# this creates an object of that would color the mask
color_mapper = Color_Mask(12)

# Need to change this to return img1, img2, affine2_to_1, mask_img1, shadow_mask1, shadow_mask2
#dataloader = DataLoader(dataset=ShadowAndMaskDataset(h, w, use_random_scale=False, use_random_affine=True),
#                        batch_size=batch_sz, shuffle=True, drop_last=True)
# Dataloader for coco
train_data_dir = 'data/train2017'
train_coco = 'data/instances_train2017.json'
#classes_path = 'data/COCO/CocoStuff164k/curated/train2017/Coco164kFew_Stuff_3.txt'
#dataL = CocoDataloader(root=train_data_dir, annotation=train_coco, input_sz=input_sz, classes_path=None)
dataL = CocoDataloader(input_sz)
dataloader = DataLoader(dataset=dataL, batch_size=batch_sz, shuffle=True, drop_last=True) # for coco add collate
curr = 0

for epoch in range(0, num_epochs):
    print("Starting epoch: %d " % (epoch))

    avg_loss = 0.  # over heads and head_epochs (and sub_heads)
    avg_acc = 0.0
    avg_ssm_loss = 0.
    # avg_loss_no_lamb = 0.
    avg_loss_count = 0
    avg_acc_count = 0

    for idx, data in enumerate(dataloader):
        # img1 is image containing shadow, img2 is transformation of img1,
        # affine2_to_1 allows reversing affine transforms to make img2 align pixels with img1,
        # mask_img1 allows zeroing out pixels that are not comparable
        #img1, img2, affine2_to_1, mask_img1 = data
        img1, img2, affine2_to_1, mask_img1, shadow_mask1 = data

        # just moving everything to cuda
        img1 = img1.cuda()
        img2 = img2.cuda()
        affine2_to_1 = affine2_to_1.cuda()
        mask_img1 = mask_img1.cuda()
        shadow_mask1 = shadow_mask1.cuda()

        net.zero_grad()

        x1_outs = net(img1)
        x2_outs = net(img2)

        # batch is passed through each subhead to calculate loss, store average loss per sub_head
        avg_loss_batch = None
        avg_loss_no_lamb_batch = None
        ssm_loss = None

	#shadow_mask1.argmax(axis=1).long() # i need to verify this for coco
        shadow_mask1_flat = shadow_mask1.view(batch_sz, input_sz, input_sz).long()
        loss, loss_no_lamb = loss_fn(x1_outs, x2_outs,
                all_affine2_to_1=affine2_to_1,
                all_mask_img1=mask_img1, lamb=lamb,
                half_T_side_dense=half_T_side_dense,
                half_T_side_sparse_min=half_T_side_sparse_min,
                half_T_side_sparse_max=half_T_side_sparse_max)

        if avg_loss_batch is None:
            avg_loss_batch = loss
        else:
            avg_loss_batch -= loss # i change this to -= to test

        # this is for accuracy
        flat_preds = torch.argmax(x1_outs.cpu().detach(), dim=1).flatten()
        flat_targets = shadow_mask1.clone().cpu().detach().flatten()

        print("This are the original shapes")
        print(x1_outs.shape, shadow_mask1.shape)
        print("This are the flat shapes")
        print(flat_preds.shape, flat_targets.shape)

        train_acc = eval_acc(flat_preds, flat_targets)
        avg_acc += train_acc
        avg_acc_count += 1

        if not np.isfinite(avg_loss_batch.item()):
            print("Loss is not finite... %s:" % str(avg_loss_batch))
            exit(1)

        avg_loss += avg_loss_batch.item()
        avg_loss_count += 1

        if use_supervised:
            loss_total = - avg_loss_batch + ssm_loss
        else:
            loss_total = avg_loss_batch

        # saving loss and accuracy
        writer.add_scalar('loss/discrete_loss', avg_loss_batch.item(), curr)
        writer.add_scalar('accuracy/discrete_acc', train_acc, curr)

        loss_total.backward()
        optimiser.step()

        # visualize outputs of first image in dataset every 10 epochs
        if curr % 500 == 0:
            print('x shape', x1_outs[0].shape)
            img_to_board = torch.argmax(x1_outs[0].cpu().detach(), dim=1).numpy()  # gets black and white image
            print('img to board shape', img_to_board.shape)
            color = color_mapper.add_color(img_to_board) # this is where we send the mask to the scrip
            writer.add_image('image/val_mask', color, curr)
            writer.add_image('image/original', img1[0], curr)
            writer.add_image('image/transformed', img2[0], curr)
            exit()

            # this saves the model
            torch.save(net.state_dict(), "saved_models/iic_e{}_{}.model".format(epoch, time_begin))

        torch.cuda.empty_cache()
        curr += 1

    # updates the learning rate
    lr *= (1 / (1 + decay * epoch))
    for param_group in optimiser.param_groups:
        param_group['lr'] = lr

    avg_loss = float(avg_loss / avg_loss_count)
    avg_acc = float(avg_acc / avg_acc_count)

    writer.add_scalar('accuracy/avg_acc', avg_acc, epoch)
    writer.add_scalar('loss/avg_loss', avg_loss, epoch)

#    if avg_loss < min_val_loss:
#        epochs_no_improve = 0
#        min_val_loss = avg_loss
#    else:
#        epochs_no_improve += 1
#    if epochs_no_improve == n_epochs_stop:
#        print("Early Stopping")
#        break

writer.close()
