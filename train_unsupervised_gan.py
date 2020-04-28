import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np

# scripts
from image_loader import ShadowShadowFreeDataset
from net10a_twohead import SegmentationNet10a
from IIC_Losses import IID_segmentation_loss
from models_for_gan import Discriminator_sf, Generator_sf

REAL = 1
FAKE = 0

h, w, in_channels = 240, 240, 3

# Lists to keep track of progress
img_list = []
# G_losses = []
# D_losses = []
# iters = 0
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
IIC = SegmentationNet10a(num_sub_heads).cuda()
Gen = Generator_sf().cuda()
# Disc = Discriminator_sf().cuda()  # use IIC as discriminator

# Initialize IIC objective function
loss_fn = IID_segmentation_loss
criterion_sf_data = torch.nn.L1Loss()
criterion_d = torch.nn.NLLLoss()  # use 2d? use log(output of softmax layer) to make cross-entropy loss

# Setup Adam optimizers for both
optimizer_iic = torch.optim.Adam(IIC.parameters(), lr=lr, betas=(beta1, 0.1))
optimizer_g = torch.optim.Adam([{'params': IIC.parameters()}, {'params': Gen.parameters()}], lr=lr, betas=(beta1, 0.1))
optimizer_d = torch.optim.Adam(IIC.parameters(), lr=lr, betas=(beta1, 0.1))

# loads images with shadows (and transformed img and affine_2_to_1 and mask) and shadow-free images
dataloader = DataLoader(dataset=ShadowShadowFreeDataset(h, w, use_random_scale=False, use_random_affine=True),
                        batch_size=batch_sz, shuffle=True, drop_last=True)

for epoch in range(0, num_epochs):
    print("Starting epoch: %d " % (epoch))

    # avg_loss = 0.  # over heads and head_epochs (and sub_heads)
    # avg_loss_no_lamb = 0.
    # avg_loss_count = 0

    for idx, data in enumerate(dataloader):
        # img1 is image containing shadow, img2 is transformation of img1,
        # affine2_to_1 allows reversing affine transforms to make img2 align pixels with img1,
        # mask_img1 allows zeroing out pixels that are not comparable
        img1, img2, affine2_to_1, mask_img1, sf_img = data

        # just moving everything to cuda
        img1 = img1.cuda()
        img2 = img2.cuda()
        affine2_to_1 = affine2_to_1.cuda()
        mask_img1 = mask_img1.cuda()

        # IIC.zero_grad()

        x1_outs = IIC(img1)
        x2_outs = IIC(img2)

        # batch is passed through each subhead to calculate loss, store average loss per sub_head
        avg_loss_batch = None
        avg_loss_no_lamb_batch = None

        for i in range(num_sub_heads):
            loss, loss_no_lamb = loss_fn(x1_outs[i], x2_outs[i],
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
        #avg_loss_batch *= -1 # this is to make the loss positive, only flip the labels
        avg_loss_no_lamb_batch /= num_sub_heads

        # track losses
        print("epoch {} average_loss {} ave_loss_no_lamb {}".format(epoch, avg_loss_batch.item(), avg_loss_no_lamb_batch.item()))

        if not np.isfinite(avg_loss_batch.item()):
            print("Loss is not finite... %s:" % str(avg_loss_batch))
            exit(1)

        # avg_loss += avg_loss_batch.item()
        # avg_loss_no_lamb += avg_loss_no_lamb_batch.item()
        # avg_loss_count += 1

        # predict shadow free image
        gen_input = torch.cat([img1, x1_outs], dim=1)  # need to double check dim 1 is correct for both
        sf_pred = Gen(gen_input)
        sf_data_loss = criterion_sf_data(sf_pred, sf_img)

        # use IIC as discriminator
        sf_mask_pred = IIC(sf_pred)
        sf_mask_pred_d = IIC(sf_pred.detach())  # for training IIC during this part (probably won't train it here?)

        # make tensor to represent perfect prediction of no shadow
        s_layer = torch.full((sf_mask_pred[0].shape), 0).cuda()
        ns_layer = torch.full((sf_mask_pred[0].shape), 1).cuda()
        no_shadow = torch.cat([ns_layer, s_layer], dim=1)

        # detach for training discriminator, but not for training generator
        disc_loss = criterion_d(torch.log(sf_mask_pred_d), no_shadow.argmax(dim=1))
        gen_adv_loss = criterion_d(torch.log(sf_mask_pred), no_shadow.argmax(dim=1))

        # during gen training IIC loss and gen data loss and gen adversarial loss all help same tasks
        gen_loss = avg_loss_batch + sf_data_loss + gen_adv_loss

        train_iic_only = False  # use for pretraining for some # of epochs if necessary
        train_gen = True
        train_disc = False  # would train IIC to predict no shadow, so we never want this True (delete when sure)

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

        # visualize outputs of last image in dataset every 10 epochs
        # if epoch % 10 == 0:
        #     o = transforms.ToPILImage()(img1[0].cpu().detach())
        #     o.save("img_visual_checks/test_img1_e{}.png".format(epoch))
        #     o = transforms.ToPILImage()(img2[0].cpu().detach())
        #     o.save("img_visual_checks/test_img2_e{}.png".format(epoch))
        #     shadow_mask1_pred_bw = torch.argmax(x1_outs[0].cpu().detach(), dim=1).numpy()  # gets black and white image
        #     cv2.imwrite('img_visual_checks/test_mask1_bw_e{}.png'.format(epoch), shadow_mask1_pred_bw[0] * 255)
        #     shadow_mask1_pred_grey = x1_outs[0][1].cpu().detach().numpy()  # gets probability pixel is black
        #     cv2.imwrite('img_visual_checks/test_mask1_grey_e{}.png'.format(epoch), shadow_mask1_pred_grey[0] * 255)
        #
        #     # this saves the model
        #     torch.save(IIC.state_dict(), "models/iic_e{}.model".format(epoch))

        torch.cuda.empty_cache()


    # avg_loss = float(avg_loss / avg_loss_count)
    # avg_loss_no_lamb = float(avg_loss_no_lamb / avg_loss_count)
