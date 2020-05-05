import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os

# scripts
from image_loader import ShadowDataset, ShadowAndMaskDataset
from coco_dataloader import CocoDataloader
from net10a_twohead import SegmentationNet10a
from IIC_Losses import IID_segmentation_loss, IID_segmentation_loss_uncollapsed
from IIC_Network import net


h, w, in_channels = 240, 240, 3

# Lists to keep track of progress
discrete_losses = []
ave_losses = []

# keep track of folders and saved files when model is run multiple times
time_begin = str(datetime.now()).replace(' ', '-')
os.mkdir("img_visual_checks/"+time_begin)
os.mkdir("loss_csvs/"+time_begin)

lamb = 1.0  # will make loss equal to loss_no_lamb
batch_sz = 8
num_sub_heads = 1
half_T_side_dense = 0
half_T_side_sparse_min = 0
half_T_side_sparse_max = 0

# Defining the learning rate, number of epochs and beta for the optimizers
lr = 0.001
beta1 = 0.5
num_epochs = 100
decay = 0.1
n_epochs_stop = 10
epochs_no_improve = 0
min_val_loss = np.Inf
use_supervised = False# set to epoch < num or similar condition?

# Create the models
net = SegmentationNet10a(num_sub_heads)
#net = net()
net.cuda()

# Initialize IIC objective function
#loss_fn = IID_segmentation_loss
loss_fn = IID_segmentation_loss_uncollapsed
criterion_ssm = torch.nn.NLLLoss()  # supervised shadow mask loss function

# Setup Adam optimizers for both
optimiser = torch.optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.1))

# Need to change this to return img1, img2, affine2_to_1, mask_img1, shadow_mask1, shadow_mask2
#dataloader = DataLoader(dataset=ShadowAndMaskDataset(h, w, use_random_scale=False, use_random_affine=True),
#                        batch_size=batch_sz, shuffle=True, drop_last=True)
# Dataloader for coco
train_data_dir = 'data/val2017'
train_coco = 'data/instances_val2017.json'
dataloader = DataLoader(dataset=CocoDataloader(root=train_data_dir, annotation=train_coco, input_sz=h),
                        batch_size=batch_sz, shuffle=True, collate_fn=CocoDataloader.collate_fn, drop_last=True)

for epoch in range(0, num_epochs):
    print("Starting epoch: %d " % (epoch))

    avg_loss = 0.  # over heads and head_epochs (and sub_heads)
    avg_ssm_loss = 0.
    # avg_loss_no_lamb = 0.
    avg_loss_count = 0

    for idx, data in enumerate(dataloader):
        # img1 is image containing shadow, img2 is transformation of img1,
        # affine2_to_1 allows reversing affine transforms to make img2 align pixels with img1,
        # mask_img1 allows zeroing out pixels that are not comparable
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


        shadow_mask1_flat = shadow_mask1.long() #shadow_mask1.argmax(axis=1).long()
        for i in range(num_sub_heads):
            loss, loss_no_lamb = loss_fn(x1_outs[i], x2_outs[i],
                    all_affine2_to_1=affine2_to_1,
                    all_mask_img1=mask_img1, lamb=lamb,
                    half_T_side_dense=half_T_side_dense,
                    half_T_side_sparse_min=half_T_side_sparse_min,
                    half_T_side_sparse_max=half_T_side_sparse_max)

            if avg_loss_batch is None:
                avg_loss_batch = loss
                # avg_loss_no_lamb_batch = loss_no_lamb
                ssm_loss = criterion_ssm(torch.log(x1_outs[i]), shadow_mask1_flat) # + criterion_ssm(torch.log(x2_outs[i]), shadow_mask2)
            else:
                avg_loss_batch += loss
                # avg_loss_no_lamb_batch += loss_no_lamb

                # assumes shadow_mask1 is tensor of 0s and 1s corresponding to argmax of x1_outs (what NLLLoss expects)
                ssm_loss += criterion_ssm(torch.log(x1_outs[i]), shadow_mask1_flat) # + criterion_ssm(torch.log(x2_outs[i]), shadow_mask2)

        avg_loss_batch /= num_sub_heads

        if idx % 10 == 0:
            discrete_losses.append([avg_loss_batch.item(), ssm_loss.item()])  # store for graphing

        if not np.isfinite(avg_loss_batch.item()):
            print("Loss is not finite... %s:" % str(avg_loss_batch))
            exit(1)

        avg_loss += avg_loss_batch.item()
        avg_ssm_loss += ssm_loss.item()
        # avg_loss_no_lamb += avg_loss_no_lamb_batch.item()
        avg_loss_count += 1

        if use_supervised:
            loss_total = - avg_loss_batch + ssm_loss
        else:
            loss_total = avg_loss_batch

        loss_total.backward()
        optimiser.step()


        # visualize outputs of first image in dataset every 10 epochs
        if epoch % 10 == 0 and idx == 0:
            o = transforms.ToPILImage()(img1[0].cpu().detach())
            o.save("img_visual_checks/"+time_begin+"/test_img1_e{}.png".format(epoch))
            o = transforms.ToPILImage()(img2[0].cpu().detach())
            o.save("img_visual_checks/"+time_begin+"/test_img2_e{}.png".format(epoch))
            shadow_mask1_pred_bw = torch.argmax(x1_outs[0].cpu().detach(), dim=1).numpy()  # gets black and white image
            cv2.imwrite("img_visual_checks/"+time_begin+"/test_mask1_bw_e{}.png".format(epoch), shadow_mask1_pred_bw[0] * 255)
            shadow_mask1_pred_grey = x1_outs[0][1].cpu().detach().numpy()  # gets probability pixel is black
            cv2.imwrite("img_visual_checks/"+time_begin+"/test_mask1_grey_e{}.png".format(epoch), shadow_mask1_pred_grey[0] * 255)

            # this saves the model
            torch.save(net.state_dict(), "saved_models/iic_e{}_{}.model".format(epoch, time_begin))

        torch.cuda.empty_cache()

    # updates the learning rate
    lr *= (1 / (1 + decay * epoch))
    for param_group in optimiser.param_groups:
        param_group['lr'] = lr

    avg_loss = float(avg_loss / avg_loss_count)
    avg_ssm_loss = float(avg_ssm_loss / avg_loss_count)
    ave_losses.append([avg_loss, avg_ssm_loss])
    print("epoch {} average_loss {} ".format(epoch, avg_loss))
    # avg_loss_no_lamb = float(avg_loss_no_lamb / avg_loss_count)

    # save lists of losses as csv files for reading and graphing later
    df1 = pd.DataFrame(list(zip(*ave_losses))).add_prefix('Col')
    filename = 'loss_csvs/' + time_begin + '/iic_ave_e' + str(epoch) + '_' + time_begin + '.csv'
    print('saving to', filename)
    df1.to_csv(filename, index=False)

    df2 = pd.DataFrame(list(zip(*discrete_losses))).add_prefix('Col')
    filename = 'loss_csvs/' + time_begin + '/iic_discrete_e' + str(epoch) + '_' + time_begin + '.csv'
    print('saving to', filename)
    df2.to_csv(filename, index=False)

#    if avg_loss < min_val_loss:
#        epochs_no_improve = 0
#        min_val_loss = avg_loss
#    else:
#        epochs_no_improve += 1
#    if epochs_no_improve == n_epochs_stop:
#        print("Early Stopping")
#        break
