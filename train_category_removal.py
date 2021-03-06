import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from random import randint

# scripts
from image_loader import ShadowShadowFreeDataset
from net10a_twohead import SegmentationNet10a, SegmentationNet10aTwoHead
from IIC_Losses import IID_segmentation_loss
from models_for_gan import Discriminator_inpainted, Generator_inpaint
from utils import remove_catx, remove_random_region, custom_loss_iic
# from coco_dataloader import CocoDataloader
from coco_dataloader_mask import CocoDataloader
from image_loader_cityscapes import CityscapesLoader

from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf

from eval import eval_acc
from color_segmentation import Color_Mask

NUM_CLASSES = 4  # number of segmentation classes in dataset, 11 for cocothings, 20 for cityscapes
REAL = 1
FAKE = 0

h, w, in_channels = 240, 240, 4
input_sz = h

# Lists to keep track of progress
discrete_losses = []
ave_losses = []

val_discrete_losses = []
val_ave_losses = []
val_ave_acc = []

# keep track of folders and saved files when model is run multiple times
time_begin = str(datetime.now()).replace(' ', '-')
# os.mkdir("img_visual_checks/" + time_begin)  # somehow no permission now

board = "boards/" + time_begin
os.mkdir(board)

writer = SummaryWriter(board)

lamb = 1.0  # will make loss equal to loss_no_lamb
batch_sz = 1
ACCUMULATION_STEPS = 10  # effectively makes batch size ACCUMULATION_STEPS * batch_sz
num_sub_heads = 1
half_T_side_dense = 0
half_T_side_sparse_min = 0
half_T_side_sparse_max = 0

curr = -1  # this is to correctly store values

# Defining the learning rate, number of epochs and beta for the optimizers
lr = 0.001
beta1 = 0.5
num_epochs = 100
decay = 0.1
n_epochs_stop = 10
epochs_no_improve = 0
min_val_loss = np.Inf
total_train = 0
Lcorrect_train = 0

'''
pre-trained model (555) config: Namespace(aff_max_rot=30.0, aff_max_scale=1.2, aff_max_shear=10.0, aff_min_rot=-30.0, 
aff_min_scale=0.8, aff_min_shear=-10.0, arch='SegmentationNet10aTwoHead', batch_sz=120, batchnorm_track=True, 
coco_164k_curated_version=6, dataloader_batch_sz=120, dataset='Coco164kCuratedFew', 
dataset_root='/scratch/local/ssd/xuji/COCO/CocoStuff164k' 
'''
# pretrained model path
pretrained_555_path = './pretrained_models/models/555/best_net.pytorch'
pretrained_555 = torch.load(pretrained_555_path)

# Create the models
# IIC = SegmentationNet10a(num_sub_heads, NUM_CLASSES).cuda()  # produces segmentation maps from images
# pretrained model should match config.arch which is 'SegmentationNet10aTwoHead'
IIC = SegmentationNet10aTwoHead()
IIC.load_state_dict(pretrained_555['net'])
IIC.cuda()
Gen = Generator_inpaint().cuda()  # fills in image with blacked out regions (catX and random area), with L1 loss for non-catX pixels
Disc = Discriminator_inpainted().cuda()  # given real image - catX and generated - catX
# use another disc? Somehow make sure catX pixels look real
# would comparing original images and generated encourage regenerating catX? Maybe use after pretraining with others?

# Initialize IIC objective function
iic_loss = IID_segmentation_loss
criterion_g_data = torch.nn.L1Loss()
criterion_d = torch.nn.BCELoss()
criterion_iic_d = torch.nn.L1Loss()

# Set up Adam optimizers
optimizer_iic = torch.optim.Adam(IIC.parameters(), lr=lr, betas=(beta1, 0.1))
optimizer_g = torch.optim.Adam(Gen.parameters(), lr=lr, betas=(beta1, 0.1))
optimizer_d = torch.optim.Adam(Disc.parameters(), lr=lr, betas=(beta1, 0.1))

# this creates an object of that would color the mask
color_mapper = Color_Mask(NUM_CLASSES)

# switch to just load img1, img2, affine2_to_1, mask as is used in IIC paper
# dataloader = DataLoader(dataset=ShadowShadowFreeDataset(h, w, use_random_scale=False, use_random_affine=True),
#                         batch_size=batch_sz, shuffle=True, drop_last=True)

coco = True
if coco:
    # Dataloader for coco
    train_data_dir = 'data/train2017'
    train_coco = 'data/instances_train2017.json'
    # train_dataloader = DataLoader(dataset=CocoDataloader(root=train_data_dir, annotation=train_coco, input_sz=input_sz, classes_path=None),
    #                         batch_size=batch_sz, shuffle=True, collate_fn=CocoDataloader.collate_fn, drop_last=True)
    dataL = CocoDataloader(input_sz, mode="train")
    train_dataloader = DataLoader(dataset=dataL, batch_size=batch_sz, shuffle=True,
                                  drop_last=True)  # for coco add collate

    val_data_dir = 'data/val2017'
    val_coco = 'data/instances_val2017.json'

    dataL = CocoDataloader(input_sz, mode="val")
    val_dataloader = DataLoader(dataset=dataL, batch_size=batch_sz, shuffle=True,
                                drop_last=True)  # for coco add collate
    # there is no option for using validation set yet
    # val_dataloader = DataLoader(dataset=CocoDataloader(root=train_data_dir, annotation=train_coco, input_sz=input_sz, classes_path=None),
    #                         batch_size=batch_sz, shuffle=True, collate_fn=CocoDataloader.collate_fn, drop_last=True)

    predict_seg = True  # False if just using ground truth segs to test rest of network

cityscapes = False
if cityscapes:
    # Use Cityscapes dataset until coco issues are resolved
    train_dataset = CityscapesLoader('train')
    # train_sampler = torch.utils.data.RandomSampler(train_dataset)
    # train_data_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    sampler595 = torch.utils.data.SubsetRandomSampler(range(0, 595))  # 1/5 the 2975 train images
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, sampler=sampler595)

    val_dataset = CityscapesLoader('val')
    # val_sampler = torch.utils.data.RandomSampler(val_dataset)
    # val_data_loader = Data.DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
    sampler100 = torch.utils.data.SubsetRandomSampler(range(0, 100))  # 1/5 the 500 val images
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_sz, sampler=sampler100)

    predict_seg = False  # False if just using ground truth segs to test rest of network

for epoch in range(0, num_epochs):
    print("Starting epoch: %d " % (epoch))

    # for mode in ['train', 'val']:
    for mode in ['train', 'val']:  # since val set is not loadable yet
        if mode == 'train':
            torch.set_grad_enabled(True)
            dataloader = train_dataloader
            IIC.train()
            Gen.train()
            Disc.train()
        elif mode == 'val':
            torch.set_grad_enabled(False)
            dataloader = val_dataloader
            IIC.eval()
            Gen.eval()
            Disc.eval()

        avg_loss = 0.  # over heads and head_epochs (and sub_heads) per epoch
        avg_acc = 0.
        avg_acc_count = 0.
        avg_disc_loss = 0.
        avg_gen_loss = 0.
        avg_sf_data_loss = 0.
        avg_gen_adv_loss = 0.
        avg_adv_seg_loss = 0.
        # avg_loss_no_lamb = 0.
        avg_loss_count = 0
        train_acc = 0

        for idx, data in enumerate(dataloader):
            # img1 is image containing shadow, img2 is transformation of img1,
            # affine2_to_1 allows reversing affine transforms to make img2 align pixels with img1,
            # mask_img1 allows zeroing out pixels that are not comparable
            # img1, img2, affine2_to_1, mask_img1 = data

            if predict_seg:
                img1, img2, affine2_to_1, mask_img1, shadow_mask1 = data  # why does CocoDataloader return 5th variable?

                # just moving everything to cuda
                img1 = img1.cuda()
                img2 = img2.cuda()
                affine2_to_1 = affine2_to_1.cuda()
                mask_img1 = mask_img1.cuda()

                x1_outs = IIC(img1)
                x2_outs = IIC(img2)

                curr += 1

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
                # avg_loss_no_lamb_batch /= num_sub_heads

                pred = torch.argmax(x1_outs[0].cpu().detach(), dim=1)
                # with tf.name_scope("input_reshape"):
                #    tf.summary.image("images", pred, epoch)
                # exit()

                if not np.isfinite(avg_loss_batch.item()):
                    print("Loss is not finite... %s:" % str(avg_loss_batch))
                    exit(1)

                avg_loss += avg_loss_batch.item()
                avg_loss_count += 1

                # assert torch.is_tensor(img1)
                # assert torch.is_tensor(x1_outs[0])  # x1_outs is list of tensors from each subhead

            # use ground truth segmentation in place of both predictions
            else:
                img, seg = data
                img1 = img.cuda()
                img2 = img.clone().cuda()
                x1_outs = [torch.zeros([1, 1, 1, 1])]  # size doesn't matter since will be replaced
                x2_outs = [torch.zeros([1, 1, 1, 1])]
                x1_outs[0] = seg.cuda()
                x2_outs[0] = seg.clone().cuda()

            # this is for accuracy
            flat_preds = torch.argmax(x1_outs[0].cpu().detach(), dim=1).flatten()
            flat_targets = shadow_mask1.clone().cpu().detach().flatten()

            train_acc = eval_acc(flat_preds, flat_targets)
            avg_acc += train_acc
            avg_acc_count += 1

            catx = randint(0, NUM_CLASSES - 1)

            # transform img1 and img2 based on segmentation output (black out catx)
            img1_no_catx = remove_catx(img1, x1_outs[0], catx)  # only one head is trained, but only one head in network
            img2_no_catx = remove_catx(img2, x2_outs[0], catx)

            # remove additional region from images
            img1_blacked = remove_random_region(img1_no_catx)
            img2_blacked = remove_random_region(img2_no_catx)

            # inpaint blacked out regions with Gen
            img1_filled = Gen(img1_blacked)
            img2_filled = Gen(img2_blacked)

            # re-blackout catX so discriminator only looks at additional region inpainting
            img1_filled_rb = remove_catx(img1_filled, x1_outs[0], catx)
            img2_filled_rb = remove_catx(img2_filled, x2_outs[0], catx)

            # catx is 0s in both, so L1 loss will not consider/effect catx regions
            filled_data_loss = criterion_g_data(img1_filled_rb, img1_no_catx) + criterion_g_data(img2_filled_rb,
                                                                                                 img2_no_catx)

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
            # with torch.no_grad:  # need gradients to be passed backwards, but parameters not updated
            xx1_outs = IIC(img1_filled)
            xx2_outs = IIC(img2_filled)
            adv_seg_loss = custom_loss_iic(xx1_outs[0], catx, criterion_iic_d) + \
                           custom_loss_iic(xx2_outs[0], catx, criterion_iic_d)

            # loss for Gwn only, not IIC
            gen_loss = filled_data_loss + gen_adv_loss + adv_seg_loss

            # calculate average losses
            avg_loss_count += 1
            # avg_loss += avg_loss_batch.item() # not using IIC,so not calculated
            avg_disc_loss += disc_loss.item()
            avg_gen_loss += gen_loss.item()
            avg_sf_data_loss += filled_data_loss.item()  # never used in backward() call since supervised
            avg_gen_adv_loss += gen_adv_loss.item()
            avg_adv_seg_loss += adv_seg_loss.item()

            if mode == 'val':
                # this consecutive lines are for the image on tensorboard
                if curr % 500 == 0:
                    img_to_board = torch.argmax(x1_outs[0].cpu().detach(), dim=1).numpy()  # gets black and white image
                    color = color_mapper.add_color(img_to_board)  # this is where we send the mask to the scrip
                    img2_to_board = img1_filled[0].cpu().detach()
                    o = img1[0].cpu().detach()
                    writer.add_image('images/val_original', o, curr)
                    writer.add_image('images/val_images', color, curr)
                    writer.add_image('images/val_mask', img_to_board, curr)
                    writer.add_image('images/val_images_filled', img2_to_board, curr)

                if idx % 10 == 0:
                    # switch back if using iic
                    # val_discrete_losses.append(
                    #     [avg_loss_batch.item(), disc_loss.item(), gen_loss.item(), filled_data_loss.item(),
                    #      gen_adv_loss.item(), adv_seg_loss.item()])  # store for graphing

                    writer.add_scalar('accuracy/discrete_acc_validation', train_acc, curr)
                    writer.add_scalar('loss/discrete_loss_validation', disc_loss.item(), curr)
                    writer.add_scalar('loss/discrete_loss_gen_validation', gen_loss.item(), curr)
                    writer.add_scalar('loss/discrete_loss_filled_data_validation', filled_data_loss.item(), curr)
                    writer.add_scalar('loss/discrete_loss_gen_adv_validation', gen_adv_loss.item(), curr)
                    writer.add_scalar('loss/discrete_loss_adv_seg_validation', adv_seg_loss.item(), curr)

            elif mode == 'train':
                # this consecutive lines are for the image on tensorboard
                if curr % 500 == 0:
                    img_to_board = torch.argmax(x1_outs[0].cpu().detach(), dim=1).numpy()  # gets black and white image
                    print(x1_outs[0].shape)
                    print(img_to_board.shape)
                    exit()
                    color = color_mapper.add_color(img_to_board)  # this is where we send the mask to the scrip
                    img2_to_board = img1_filled[0].cpu().detach()
                    o = img1[0].cpu().detach()
                    writer.add_image('images/train_original', o, curr)
                    writer.add_image('images/train_images', color, curr)
                    writer.add_image('images/train_images_filled', img2_to_board, curr)

                if idx % 10 == 0:
                    # switch back if using iic
                    # discrete_losses.append(
                    #     [avg_loss_batch.item(), disc_loss.item(), gen_loss.item(), filled_data_loss.item(),
                    #      gen_adv_loss.item(), adv_seg_loss.item()])  # store for graphing

                    writer.add_scalar('accuracy/discrete_acc_train', train_acc, curr)
                    writer.add_scalar('loss/discrete_loss_train', disc_loss.item(), curr)
                    writer.add_scalar('loss/discrete_loss_gen_train', gen_loss.item(), curr)
                    writer.add_scalar('loss/discrete_loss_filled_data_train', filled_data_loss.item(), curr)
                    writer.add_scalar('loss/discrete_loss_gen_adv_train', gen_adv_loss.item(), curr)
                    writer.add_scalar('loss/discrete_loss_adv_seg_train', adv_seg_loss.item(), curr)

                # if epoch < 50:
                if epoch < 1 and idx < 3000 and predict_seg:  # pretrain just iic (later just load pretrained one instead)
                    train_iic_only = True  # use for pretraining for some # of epochs if necessary
                    train_gen = False
                    train_disc = False
                elif predict_seg:
                    train_iic_only = True
                    train_gen = True
                    train_disc = True
                else:
                    train_iic_only = False  # never train iic since not using (no predict_seg)
                    train_gen = True
                    train_disc = True

                if train_iic_only:
                    # optimizer_iic.zero_grad()
                    avg_loss_batch.backward()
                    if (idx + 1) % ACCUMULATION_STEPS == 0:
                        optimizer_iic.step()  # important that this optimizer steps before adv_seg_loss.backward() is called
                        optimizer_iic.zero_grad()

                if train_gen:
                    # optimizer_g.zero_grad()
                    gen_loss.backward()
                    if (idx + 1) % ACCUMULATION_STEPS == 0:
                        optimizer_g.step()  # only includes Gen params, not IIC params
                        optimizer_g.zero_grad()

                if train_disc:
                    # optimizer_d.zero_grad()
                    disc_loss.backward()
                    if (idx + 1) % ACCUMULATION_STEPS == 0:
                        optimizer_d.step()
                        optimizer_d.zero_grad()

                if idx % 1000 == 0:
                    torch.save({
                        'epoch': epoch,
                        'idx': idx,
                        'time_begin': time_begin,
                        'IIC_state_dict': IIC.state_dict(),
                        'Gen_state_dict': Gen.state_dict(),
                        'Disc_state_dict': Disc.state_dict(),
                    }, "saved_models/cat_removal_e{}_idx{}_{}.model".format(epoch, idx, time_begin))

        torch.cuda.empty_cache()
        # change to make loop only go through portion of dataset since there are so many training files
        # validation set only needed for after IIC is trained alone (maximizing mutual info will not overfit training data)

    # updates the learning rate
    lr *= (1 / (1 + decay * epoch))
    for param_group in optimizer_iic.param_groups:
        param_group['lr'] = lr

    # calculate average over epoch
    avg_loss = float(avg_loss / avg_loss_count)
    avg_acc = float(avg_acc / avg_loss_count)
    avg_disc_loss = float(avg_disc_loss / avg_loss_count)
    avg_gen_loss = float(avg_gen_loss / avg_loss_count)
    avg_sf_data_loss = float(avg_sf_data_loss / avg_loss_count)
    avg_gen_adv_loss = float(avg_gen_adv_loss / avg_loss_count)
    avg_adv_seg_loss = float(avg_adv_seg_loss / avg_loss_count)

    if mode == 'train':
        # if not predict_seg:
        #    train_acc = 100  # since not using iic
        ## keep track of accuracy to plot
        # ave_acc.append([train_acc])

        writer.add_scalar('accuracy/avg_acc_train', avg_acc, epoch)
        writer.add_scalar('loss/avg_loss_train', avg_loss, epoch)
        writer.add_scalar('loss/avg_loss_disc_train', avg_disc_loss, epoch)
        writer.add_scalar('loss/avg_loss_gen_train', avg_gen_loss, epoch)
        writer.add_scalar('loss/avg_loss_sf_data__train', avg_sf_data_loss, epoch)
        writer.add_scalar('loss/avg_loss_gen_adv_train', avg_gen_adv_loss, epoch)
        writer.add_scalar('loss/avg_loss_adv_seg_train', avg_adv_seg_loss, epoch)

        # ave_losses.append([avg_loss, avg_disc_loss, avg_gen_loss, avg_sf_data_loss, avg_gen_adv_loss,
        #                   avg_adv_seg_loss])  # store for graphing

        # save lists of losses and accuracy as csv files for reading and graphing later
        # df0 = pd.DataFrame(list(zip(*ave_acc))).add_prefix('Col')
        # filename = 'loss_csvs/' + time_begin + '/cat_removal_acc_e' + str(epoch) + '_' + time_begin + '.csv'
        # print('saving to', filename)
        # df0.to_csv(filename, index=False)

        # df1 = pd.DataFrame(list(zip(*ave_losses))).add_prefix('Col')
        # filename = 'loss_csvs/cat_removal_ave_e' + str(epoch) + '_' + time_begin + '.csv'
        # print('saving to', filename)
        # df1.to_csv(filename, index=False)

        # df2 = pd.DataFrame(list(zip(*discrete_losses))).add_prefix('Col')
        # filename = 'loss_csvs/cat_removal_discrete_e' + str(epoch) + '_' + time_begin + '.csv'
        # print('saving to', filename)
        # df2.to_csv(filename, index=False)

    elif mode == 'val':
        #        if not predict_seg:
        #            train_acc = 100  # since not using iic

        writer.add_scalar('accuracy/avg_acc_validation', avg_acc, epoch)
        writer.add_scalar('loss/avg_loss_validation', avg_loss, epoch)
        writer.add_scalar('loss/avg_loss_disc_validation', avg_disc_loss, epoch)
        writer.add_scalar('loss/avg_loss_gen_validation', avg_gen_loss, epoch)
        writer.add_scalar('loss/avg_loss_sf_data__validation', avg_sf_data_loss, epoch)
        writer.add_scalar('loss/avg_loss_gen_adv_validation', avg_gen_adv_loss, epoch)
        writer.add_scalar('loss/avg_loss_adv_seg_validation', avg_adv_seg_loss, epoch)

        # keep track of accuracy to plot
#        val_ave_acc.append([train_acc])
#
#        val_ave_losses.append([avg_loss, avg_disc_loss, avg_gen_loss, avg_sf_data_loss, avg_gen_adv_loss,
#                           avg_adv_seg_loss])  # store for graphing
#
#        # save lists of losses and accuracy as csv files for reading and graphing later
#        df0 = pd.DataFrame(list(zip(*val_ave_acc))).add_prefix('Col')
#        filename = 'loss_csvs/' + time_begin + '/cat_removal_val_acc_e' + str(epoch) + '_' + time_begin + '.csv'
#        print('saving to', filename)
#        df0.to_csv(filename, index=False)
#
#        df1 = pd.DataFrame(list(zip(*val_ave_losses))).add_prefix('Col')
#        filename = 'loss_csvs/cat_removal_val_ave_e' + str(epoch) + '_' + time_begin + '.csv'
#        print('saving to', filename)
#        df1.to_csv(filename, index=False)
#
#        df2 = pd.DataFrame(list(zip(*val_discrete_losses))).add_prefix('Col')
#        filename = 'loss_csvs/cat_removal_val_discrete_e' + str(epoch) + '_' + time_begin + '.csv'
#        print('saving to', filename)
#        df2.to_csv(filename, index=False)


writer.close()
