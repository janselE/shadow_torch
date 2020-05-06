import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime

# scripts
from image_loader import ShadowShadowFreeDataset
from net10a_twohead import SegmentationNet10a
from IIC_Losses import IID_segmentation_loss
from models_for_gan import Discriminator_inpainted, Generator_inpaint
from utils import remove_catx, remove_random_region, custom_loss_g, custom_loss_iic
from coco_dataloader import CocoDataloader

REAL = 1
FAKE = 0

h, w, in_channels = 240, 240, 3

# Lists to keep track of progress
discrete_losses = []
ave_losses = []

# keep track of folders and saved files when model is run multiple times
time_begin = str(datetime.now()).replace(' ', '-')
os.mkdir("img_visual_checks/" + time_begin)

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

# Create the models
IIC = SegmentationNet10a(num_sub_heads).cuda()  # produces segmentation maps from images
Gen = Generator_inpaint().cuda()  # fills in image with blacked out regions (catX and random area), with L1 loss for non-catX pixels
Disc = Discriminator_inpainted().cuda()  # given real image - catX and generated - catX
# use another disc? Somehow make sure catX pixels look real
# would comparing original images and generated encourage regenerating catX? Maybe use after pretraining with others?

# Initialize IIC objective function
iic_loss = IID_segmentation_loss
criterion_g_data = custom_loss_g()  # torch.nn.L1Loss()
criterion_d = torch.nn.BCELoss()

# Set up Adam optimizers
optimizer_iic = torch.optim.Adam(IIC.parameters(), lr=lr, betas=(beta1, 0.1))
optimizer_g = torch.optim.Adam(Gen.parameters(), lr=lr, betas=(beta1, 0.1))
optimizer_d = torch.optim.Adam(Disc.parameters(), lr=lr, betas=(beta1, 0.1))

# switch to just load img1, img2, affine2_to_1, mask as is used in IIC paper
# dataloader = DataLoader(dataset=ShadowShadowFreeDataset(h, w, use_random_scale=False, use_random_affine=True),
#                         batch_size=batch_sz, shuffle=True, drop_last=True)

# Dataloader for coco
train_data_dir = 'data/val2017'
train_coco = 'data/instances_val2017.json'
dataloader = DataLoader(dataset=CocoDataloader(root=train_data_dir, annotation=train_coco, input_sz=input_sz),
                        batch_size=batch_sz, shuffle=True, collate_fn=CocoDataloader.collate_fn, drop_last=True)

for epoch in range(0, num_epochs):
    print("Starting epoch: %d " % (epoch))

    avg_loss = 0.  # over heads and head_epochs (and sub_heads) per epoch
    avg_disc_loss = 0.
    avg_gen_loss = 0.
    avg_sf_data_loss = 0.
    avg_gen_adv_loss = 0.
    avg_adv_seg_loss = 0.
    # avg_loss_no_lamb = 0.
    avg_loss_count = 0

    for idx, data in enumerate(dataloader):
        # img1 is image containing shadow, img2 is transformation of img1,
        # affine2_to_1 allows reversing affine transforms to make img2 align pixels with img1,
        # mask_img1 allows zeroing out pixels that are not comparable
        # img1, img2, affine2_to_1, mask_img1 = data
        img1, img2, affine2_to_1, mask_img1, shadow_mask1 = data  # why does CocoDataloader return 5th variable?

        # just moving everything to cuda
        img1 = img1.cuda()
        img2 = img2.cuda()
        affine2_to_1 = affine2_to_1.cuda()
        mask_img1 = mask_img1.cuda()

        x1_outs = IIC(img1)
        x2_outs = IIC(img2)

        # batch is passed through each subhead to calculate loss, store average loss per sub_head
        avg_loss_batch = None
        avg_loss_no_lamb_batch = None

        for i in range(num_sub_heads):
            loss, loss_no_lamb = iic_loss(x1_outs[i], x2_outs[i],
                                          all_affine2_to_1=affine2_to_1,
                                          all_mask_img1=mask_img1, lamb=lamb,
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
        # avg_loss_batch *= -1 # this is to make the loss positive, only flip the labels
        avg_loss_no_lamb_batch /= num_sub_heads

        # track losses
        print("epoch {} average_loss {} ave_loss_no_lamb {}".format(epoch, avg_loss_batch.item(),
                                                                    avg_loss_no_lamb_batch.item()))

        if not np.isfinite(avg_loss_batch.item()):
            print("Loss is not finite... %s:" % str(avg_loss_batch))
            exit(1)

        # assert torch.is_tensor(img1)
        # assert torch.is_tensor(x1_outs[0])  # x1_outs is list of tensors from each subhead

        # transform img1 and img2 based on segmentation output (black out catx)
        img1_no_catx = remove_catx(img1)
        img2_no_catx = remove_catx(img2)

        # remove additional region from images
        img1_blacked = remove_random_region(img1_no_catx)
        img2_blacked = remove_random_region(img2_no_catx)

        # inpaint blacked out regions with Gen
        img1_filled = Gen(img1_blacked)
        img2_filled = Gen(img2_blacked)
        # need to make sure loss doesn't compare catX pixels (doesn't learn to keep them black)
        # write custom loss function?
        filled_data_loss = criterion_g_data(img1_filled, img1_no_catx) + criterion_g_data(img2_filled, img2_no_catx)

        # re-blackout catX so discriminator only looks at additional region inpainting
        img1_filled_rb = remove_catx(img1_filled)
        img2_filled_rb = remove_catx(img2_filled)

        # use discriminator
        prob_img1_nc_real = Disc(img1_no_catx.detach())
        prob_img2_nc_real = Disc(img2_no_catx.detach())
        prob_img1_rb_pred = Disc(img1_filled_rb.detach())
        prob_img2_rb_pred = Disc(img2_filled_rb.detach())
        prob_img1_rb_pred_adv = Disc(img1_filled_rb)  # not detached to backprop to Gen and IIC
        prob_img2_rb_pred_adv = Disc(img2_filled_rb)  # not detached to backprop to Gen and IIC

        # get tensors of labels to compute loss for Disc
        REAL_t = torch.full((prob_img1_nc_real.shape), REAL).cuda()  # tensor of REAL labels
        FAKE_t = torch.full((prob_img1_nc_real.shape), FAKE).cuda()  # tensor of FAKE labels

        # blacked out regions are the same in both discriminator inputs, so will not backprop or effect generator
        disc_loss = criterion_d(prob_img1_rb_pred, FAKE_t) + criterion_d(prob_img1_nc_real, REAL_t) + \
                    criterion_d(prob_img2_rb_pred, FAKE_t) + criterion_d(prob_img2_nc_real, REAL_t)
        gen_adv_loss = criterion_d(prob_img1_rb_pred_adv, REAL_t) + criterion_d(prob_img2_rb_pred_adv, REAL_t)

        # use IIC to enforce that filled images do not contain catX
        xx1_outs = IIC(img1_filled)
        xx2_outs = IIC(img2_filled)
        adv_seg_loss = custom_loss_iic(xx1_outs[i], x1_outs[i])

        # loss for Gwn only, not IIC
        gen_loss = filled_data_loss + gen_adv_loss + adv_seg_loss

        if idx % 10 == 0:
            discrete_losses.append([avg_loss_batch.item(), disc_loss.item(), gen_loss.item(), filled_data_loss.item(),
                                    gen_adv_loss.item(), adv_seg_loss.item()])  # store for graphing

        # calculate average losses
        avg_loss_count += 1
        avg_loss += avg_loss_batch.item()
        avg_disc_loss += disc_loss.item()
        avg_gen_loss += gen_loss.item()
        avg_sf_data_loss += filled_data_loss.item()  # never used in backward() call since supervised
        avg_gen_adv_loss += gen_adv_loss.item()
        avg_adv_seg_loss += adv_seg_loss.item()

        train_iic_only = False  # use for pretraining for some # of epochs if necessary
        train_gen = True
        train_disc = True

        if train_iic_only:
            optimizer_iic.zero_grad()
            avg_loss_batch.backward()
            optimizer_iic.step()

        if train_gen:
            optimizer_g.zero_grad()  # both IIC params and Gen params
            gen_loss.backward()
            optimizer_g.step()

        if train_disc:
            optimizer_d.zero_grad()
            disc_loss.backward()
            optimizer_d.step()

        # visualize outputs of first image in dataset every 10 epochs
        if epoch % 10 == 0 and idx == 0:
            # o = transforms.ToPILImage()(img1[0].cpu().detach())
            # o.save("img_visual_checks/" + time_begin + "/test_img1_e{}_{}.png".format(epoch, time_begin))
            # o = transforms.ToPILImage()(img2[0].cpu().detach())
            # o.save("img_visual_checks/" + time_begin + "/test_img2_e{}_{}.png".format(epoch, time_begin))
            # shadow_mask1_pred_bw = torch.argmax(x1_outs[0].cpu().detach(),
            #                                     dim=1).numpy()  # gets black and white image
            # cv2.imwrite("img_visual_checks/" + time_begin + "/test_mask1_bw_e{}_{}.png".format(epoch, time_begin),
            #             shadow_mask1_pred_bw[0] * 255)
            # shadow_mask1_pred_grey = x1_outs[0][1].cpu().detach().numpy()  # gets probability pixel is black
            # cv2.imwrite("img_visual_checks/" + time_begin + "/test_mask1_grey_e{}_{}.png".format(epoch, time_begin),
            #             shadow_mask1_pred_grey[0] * 255)
            #
            # # this saves the model
            torch.save(IIC.state_dict(), "saved_models/baseline_gan_e{}_{}.model".format(epoch, time_begin))

        torch.cuda.empty_cache()

    # calculate average over epoch
    avg_loss = float(avg_loss / avg_loss_count)
    avg_disc_loss = float(avg_disc_loss / avg_loss_count)
    avg_gen_loss = float(avg_gen_loss / avg_loss_count)
    avg_sf_data_loss = float(avg_sf_data_loss / avg_loss_count)
    avg_gen_adv_loss = float(avg_gen_adv_loss / avg_loss_count)
    avg_adv_seg_loss = float(avg_adv_seg_loss / avg_loss_count)

    ave_losses.append([avg_loss, avg_disc_loss, avg_gen_loss, avg_sf_data_loss, avg_gen_adv_loss, avg_adv_seg_loss])  # store for graphing

    # save lists of losses as csv files for reading and graphing later
    df1 = pd.DataFrame(list(zip(*ave_losses))).add_prefix('Col')
    filename = 'loss_csvs/baseline_gan_ave_e' + str(epoch) + '_' + time_begin + '.csv'
    print('saving to', filename)
    df1.to_csv(filename, index=False)

    df2 = pd.DataFrame(list(zip(*discrete_losses))).add_prefix('Col')
    filename = 'loss_csvs/baseline_gan_discrete_e' + str(epoch) + '_' + time_begin + '.csv'
    print('saving to', filename)
    df2.to_csv(filename, index=False)
