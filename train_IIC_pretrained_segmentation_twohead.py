from __future__ import print_function

import argparse
import itertools
import pickle
import sys
from datetime import datetime

import matplotlib
import numpy as np
import torch

# this is for tensorboard
from torch.utils.tensorboard import SummaryWriter

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

sys.path.append('IIC/code')

import IIC.code.archs as archs
from IIC.code.utils.cluster.general import config_to_str, get_opt, update_lr, nice
from IIC.code.utils.cluster.transforms import sobel_process
from IIC.code.utils.segmentation.segmentation_eval import \
    segmentation_eval
from IIC.code.utils.segmentation.IID_losses import IID_segmentation_loss, \
    IID_segmentation_loss_uncollapsed
from IIC.code.utils.segmentation.data import segmentation_create_dataloaders
from IIC.code.utils.segmentation.general import set_segmentation_input_channels

time_begin = str(datetime.now()).replace(' ', '-')

"""
  Fully unsupervised clustering for segmentation ("IIC" = "IID").
  Train and test script.
  Network has two heads, for overclustering and final clustering.
"""

# from IIC/code/scripts/segmentation/segmentation_twohead.py

# Options ----------------------------------------------------------------------
'''
parser = argparse.ArgumentParser()
parser.add_argument("--model_ind", type=int, required=True)
parser.add_argument("--arch", type=str, required=True)
parser.add_argument("--opt", type=str, default="Adam")
parser.add_argument("--mode", type=str, default="IID")  # or IID+

parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--dataset_root", type=str, required=True)

parser.add_argument("--use_coarse_labels", default=False,
                    action="store_true")  # COCO, Potsdam
parser.add_argument("--fine_to_coarse_dict", type=str,  # COCO
                    default="/users/xuji/iid/iid_private/code/datasets"
                            "/segmentation/util/out/fine_to_coarse_dict.pickle")
parser.add_argument("--include_things_labels", default=False,
                    action="store_true")  # COCO
parser.add_argument("--incl_animal_things", default=False,
                    action="store_true")  # COCO
parser.add_argument("--coco_164k_curated_version", type=int, default=-1)  # COCO

parser.add_argument("--gt_k", type=int, required=True)
parser.add_argument("--output_k_A", type=int, required=True)
parser.add_argument("--output_k_B", type=int, required=True)

parser.add_argument("--lamb_A", type=float, default=1.0)
parser.add_argument("--lamb_B", type=float, default=1.0)

parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--lr_schedule", type=int, nargs="+", default=[])
parser.add_argument("--lr_mult", type=float, default=0.1)

parser.add_argument("--use_uncollapsed_loss", default=False,
                    action="store_true")
parser.add_argument("--mask_input", default=False, action="store_true")

parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--batch_sz", type=int, required=True)  # num pairs
parser.add_argument("--num_dataloaders", type=int, default=3)
parser.add_argument("--num_sub_heads", type=int, default=5)

parser.add_argument("--out_root", type=str,
                    default="/scratch/shared/slow/xuji/iid_private")
parser.add_argument("--restart", default=False, action="store_true")

parser.add_argument("--save_freq", type=int, default=5)
parser.add_argument("--test_code", default=False, action="store_true")

parser.add_argument("--head_B_first", default=False, action="store_true")
parser.add_argument("--batchnorm_track", default=False, action="store_true")

# data transforms
parser.add_argument("--no_sobel", default=False, action="store_true")

parser.add_argument("--include_rgb", default=False, action="store_true")
parser.add_argument("--pre_scale_all", default=False,
                    action="store_true")  # new
parser.add_argument("--pre_scale_factor", type=float, default=0.5)  #

parser.add_argument("--input_sz", type=int, default=161)  # half of kazuto1011

parser.add_argument("--use_random_scale", default=False,
                    action="store_true")  # new
parser.add_argument("--scale_min", type=float, default=0.6)
parser.add_argument("--scale_max", type=float, default=1.4)

# transforms we learn invariance to
parser.add_argument("--jitter_brightness", type=float, default=0.4)
parser.add_argument("--jitter_contrast", type=float, default=0.4)
parser.add_argument("--jitter_saturation", type=float, default=0.4)
parser.add_argument("--jitter_hue", type=float, default=0.125)

parser.add_argument("--flip_p", type=float, default=0.5)

parser.add_argument("--use_random_affine", default=False,
                    action="store_true")  # new
parser.add_argument("--aff_min_rot", type=float, default=-30.)  # degrees
parser.add_argument("--aff_max_rot", type=float, default=30.)  # degrees
parser.add_argument("--aff_min_shear", type=float, default=-10.)  # degrees
parser.add_argument("--aff_max_shear", type=float, default=10.)  # degrees
parser.add_argument("--aff_min_scale", type=float, default=0.8)
parser.add_argument("--aff_max_scale", type=float, default=1.2)

# local spatial invariance. Dense means done convolutionally. Sparse means done
#  once in data augmentation phase. These are not mutually exclusive
parser.add_argument("--half_T_side_dense", type=int, default=0)
parser.add_argument("--half_T_side_sparse_min", type=int, default=0)
parser.add_argument("--half_T_side_sparse_max", type=int, default=0)
'''

board = "boards/" + time_begin
os.mkdir(board)

writer = SummaryWriter(board)

# config = parser.parse_args()  # change to have predefined args or load config file
with open('./pretrained_models/models/555/config.pickle', "rb") as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    config = u.load()

# config = pickle.load(open('./pretrained_models/models/555/config.pickle', "rb"))
# make sure --dataset_root is set to (absolute path of) my_CocoStuff164k_directory, and --fine_to_coarse_dict is set to
# (absolute path of) code/datasets/segmentation/util/out/fine_to_coarse_dict.pickle
config.dataset_root = '/work/LAS/jannesar-lab/shadow_torch/data3'
config.fine_to_coarse_dict = '/work/LAS/jannesar-lab/shadow_torch/IIC/code/datasets/segmentation/util/out/fine_to_coarse_dict.pickle'
config.out_root = "configs"
config.model_ind = 555
config.restart = True 
config.test_code = True 

'''
The config file contains the following:

Namespace(aff_max_rot=30.0, 
          aff_max_scale=1.2, 
          aff_max_shear=10.0, 
          aff_min_rot=-30.0, 
          aff_min_scale=0.8,
          aff_min_shear=-10.0, 
          arch='SegmentationNet10aTwoHead', 
          batch_sz=120, 
          batchnorm_track=True,
          coco_164k_curated_version=6, 
          dataloader_batch_sz=120, 
          dataset='Coco164kCuratedFew',
          dataset_root='/scratch/local/ssd/xuji/COCO/CocoStuff164k', 
          epoch_acc=[0.3565545631495595, 0.4611281067913882, ..., 0.7049956389275988],
          epoch_ari=[-1.0, -1.0, ..., -1.0], 
          epoch_loss_head_A=[-0.9078965678955071, ..., -1.4899453563627854], 
          epoch_loss_head_B=[-1.6547690672812119, ..., -1.7504075118918825],
          epoch_loss_no_lamb_head_A=[-0.9078965678955071, ..., 0.6537394435966716], 
          epoch_masses=array([[0.01755809, 0.1158256, 0.86661631],
                            [0.32405867, 0.32008566, 0.35585567]]),
          epoch_nmi=[-1.0, -1.0, -1.0], 
          eval_mode='hung',
          fine_to_coarse_dict='/users/xuji/iid/iid_private/code/datasets/segmentation/util/out/fine_to_coarse_dict.pickle',
          flip_p=0.5, 
          gt_k=3, 
          half_T_side_dense=10, 
          half_T_side_sparse_max=0, 
          half_T_side_sparse_min=0, 
          in_channels=5,
          incl_animal_things=False, 
          include_rgb=True, 
          include_things_labels=False, 
          input_sz=128, 
          jitter_brightness=0.4,
          jitter_contrast=0.4, 
          jitter_hue=0.125, 
          jitter_saturation=0.4, 
          lamb_A=1.0, 
          lamb_B=1.5, 
          last_epoch=108,
          lr=0.0001, 
          lr_mult=0.1, 
          lr_schedule=[], 
          mapping_assignment_partitions=['train2017', 'val2017'],
          mapping_test_partitions=['train2017', 'val2017'], 
          mask_input=False, 
          mode='IID', 
          model_ind=555,
          no_pre_eval=False, 
          no_sobel=False, 
          num_dataloaders=1, 
          num_epochs=4800, 
          num_heads=1, 
          num_sub_heads=1,
          opt='Adam', 
          out_dir='/scratch/shared/slow/xuji/iid_private/555',
          out_root='/scratch/shared/slow/xuji/iid_private', 
          output_k=3, output_k_A=15, output_k_B=3, 
          pre_scale_all=True,
          pre_scale_factor=0.33, 
          restart=True, 
          save_multiple=True, 
          scale_max=1.4, scale_min=0.6,
          train_partitions=['train2017', 'val2017'], 
          use_coarse_labels=True, 
          use_doersch_datasets=False,
          use_random_affine=False, 
          use_random_scale=False, 
          use_uncollapsed_loss=True, 
          using_IR=False)

'''


# Setup ------------------------------------------------------------------------

config.out_root = '/work/LAS/jannesar-lab/shadow_torch/saved_models'
config.out_dir = '/work/LAS/jannesar-lab/shadow_torch/saved_models/' + time_begin
os.mkdir(config.out_dir)
config.batch_sz = 1  # until we implement gradient accumulation
config.dataloader_batch_sz = int(config.batch_sz / config.num_dataloaders)  # should be 1/1
assert (config.mode == "IID")
assert ("TwoHead" in config.arch)
assert (config.output_k_B == config.gt_k)
config.output_k = config.output_k_B  # for eval code
assert (config.output_k_A >= config.gt_k)  # sanity
config.use_doersch_datasets = False
config.eval_mode = "hung"
set_segmentation_input_channels(config)  # changed config.in_channels
print('config.in_channels = ', config.in_channels)

if not os.path.exists(config.out_dir):
   os.makedirs(config.out_dir)  # did above

if config.restart:
   config_name = "config.pickle"
   dict_name = "latest.pytorch"

   given_config = config
   reloaded_config_path = os.path.join(given_config.out_dir, config_name)
   # loads config file, which we did above instead
   # print("Loading restarting config from: %s" % reloaded_config_path)
   # with open(reloaded_config_path, "rb") as config_f:
   #     config = pickle.load(config_f)
   # assert (config.model_ind == given_config.model_ind)
   # config.restart = True

   # copy over new num_epochs and lr schedule
   config.num_epochs = given_config.num_epochs
   config.lr_schedule = given_config.lr_schedule
   print("Given config: %s" % config_to_str(config))
else:
   print("Given config: %s" % config_to_str(config))

# Model ------------------------------------------------------
print("Starting the Model section")

def train():
    dataloaders_head_A, mapping_assignment_dataloader, mapping_test_dataloader = segmentation_create_dataloaders(config) 
    dataloaders_head_B = dataloaders_head_A  # unlike for clustering datasets

    net = archs.__dict__[config.arch](config)
#        if config.restart:
#            dict = torch.load(config.out_dir)
#            net.load_state_dict(dict["net"])

    # pretrained model path, should load pretrained weights
    print("Loading pretrained")
    pretrained_555_path = './pretrained_models/models/555/best_net.pytorch'
    pretrained_555 = torch.load(pretrained_555_path)
    print("Done loading pretrained")

    print("Loading state dic")
    net.load_state_dict(pretrained_555)
    print("Done loading state dic")

    print("Setting the model in cuda")
    net.cuda()
    net = torch.nn.DataParallel(net)
    net.train()
    print("Done setting the model in cuda")

    optimiser = get_opt(config.opt)(net.module.parameters(), lr=config.lr)
#    if config.restart:
#        optimiser.load_state_dict(dict["optimiser"])


    heads = ["A", "B"]
    if hasattr(config, "head_B_first") and config.head_B_first:
        heads = ["B", "A"]

    # Results
    # ----------------------------------------------------------------------
    print("Starting the Results section")

#    if config.restart:
#        next_epoch = config.last_epoch + 1
#        print("starting from epoch %d" % next_epoch)
#
#        config.epoch_acc = config.epoch_acc[:next_epoch]  # in case we overshot
#        config.epoch_avg_subhead_acc = config.epoch_avg_subhead_acc[:next_epoch]
#        config.epoch_stats = config.epoch_stats[:next_epoch]
#
#        config.epoch_loss_head_A = config.epoch_loss_head_A[:(next_epoch - 1)]
#        config.epoch_loss_no_lamb_head_A = config.epoch_loss_no_lamb_head_A[
#                                           :(next_epoch - 1)]
#        config.epoch_loss_head_B = config.epoch_loss_head_B[:(next_epoch - 1)]
#        config.epoch_loss_no_lamb_head_B = config.epoch_loss_no_lamb_head_B[
#                                           :(next_epoch - 1)]
#    else:
    config.epoch_acc = []
    config.epoch_avg_subhead_acc = []
    config.epoch_stats = []
    
    config.epoch_loss_head_A = []
    config.epoch_loss_no_lamb_head_A = []
    
    config.epoch_loss_head_B = []
    config.epoch_loss_no_lamb_head_B = []
    print("Line after all initialization")
    
    _ = segmentation_eval(config, net,
                          mapping_assignment_dataloader=mapping_assignment_dataloader,
                          mapping_test_dataloader=mapping_test_dataloader,
                          sobel=(not config.no_sobel),
                          using_IR=config.using_IR)
    
    print("Pre: time %s: \n %s" % (datetime.now(), nice(config.epoch_stats[-1])))
    sys.stdout.flush()
    next_epoch = 1

    fig, axarr = plt.subplots(6, sharex=False, figsize=(20, 20))

    if not config.use_uncollapsed_loss:
        print("using condensed loss (default)")
        loss_fn = IID_segmentation_loss
    else:
        print("using uncollapsed loss!")
        loss_fn = IID_segmentation_loss_uncollapsed

    # Train
    # ------------------------------------------------------------------------
    print("Starting the Traing section")

    for e_i in range(next_epoch, config.num_epochs):
        print("Starting e_i: %d %s" % (e_i, datetime.now()))
        exit()
        sys.stdout.flush()

        if e_i in config.lr_schedule:
            optimiser = update_lr(optimiser, lr_mult=config.lr_mult)

        for head_i in range(2):
            head = heads[head_i]
            if head == "A":
                dataloaders = dataloaders_head_A
                epoch_loss = config.epoch_loss_head_A
                epoch_loss_no_lamb = config.epoch_loss_no_lamb_head_A
                lamb = config.lamb_A

            elif head == "B":
                dataloaders = dataloaders_head_B
                epoch_loss = config.epoch_loss_head_B
                epoch_loss_no_lamb = config.epoch_loss_no_lamb_head_B
                lamb = config.lamb_B

            iterators = (d for d in dataloaders)
            b_i = 0
            avg_loss = 0.  # over heads and head_epochs (and sub_heads)
            avg_loss_no_lamb = 0.
            avg_loss_count = 0

            for tup in itertools.izip(*iterators):
                net.module.zero_grad()

                if not config.no_sobel:
                    pre_channels = config.in_channels - 1
                else:
                    pre_channels = config.in_channels

                all_img1 = torch.zeros(config.batch_sz, pre_channels,
                                       config.input_sz, config.input_sz).to(
                    torch.float32).cuda()
                all_img2 = torch.zeros(config.batch_sz, pre_channels,
                                       config.input_sz, config.input_sz).to(
                    torch.float32).cuda()
                all_affine2_to_1 = torch.zeros(config.batch_sz, 2, 3).to(
                    torch.float32).cuda()
                all_mask_img1 = torch.zeros(config.batch_sz, config.input_sz,
                                            config.input_sz).to(torch.float32).cuda()

                curr_batch_sz = tup[0][0].shape[0]
                for d_i in range(config.num_dataloaders):
                    img1, img2, affine2_to_1, mask_img1 = tup[d_i]
                    assert (img1.shape[0] == curr_batch_sz)

                    actual_batch_start = d_i * curr_batch_sz
                    actual_batch_end = actual_batch_start + curr_batch_sz

                    all_img1[actual_batch_start:actual_batch_end, :, :, :] = img1
                    all_img2[actual_batch_start:actual_batch_end, :, :, :] = img2
                    all_affine2_to_1[actual_batch_start:actual_batch_end, :,
                    :] = affine2_to_1
                    all_mask_img1[actual_batch_start:actual_batch_end, :, :] = mask_img1

                if not (curr_batch_sz == config.dataloader_batch_sz) and (
                        e_i == next_epoch):
                    print("last batch sz %d" % curr_batch_sz)

                curr_total_batch_sz = curr_batch_sz * config.num_dataloaders  # times 2
                all_img1 = all_img1[:curr_total_batch_sz, :, :, :]
                all_img2 = all_img2[:curr_total_batch_sz, :, :, :]
                all_affine2_to_1 = all_affine2_to_1[:curr_total_batch_sz, :, :]
                all_mask_img1 = all_mask_img1[:curr_total_batch_sz, :, :]

                if (not config.no_sobel):
                    all_img1 = sobel_process(all_img1, config.include_rgb,
                                             using_IR=config.using_IR)
                    all_img2 = sobel_process(all_img2, config.include_rgb,
                                             using_IR=config.using_IR)

                x1_outs = net(all_img1, head=head)
                x2_outs = net(all_img2, head=head)

                avg_loss_batch = None  # avg over the heads
                avg_loss_no_lamb_batch = None

                for i in range(config.num_sub_heads):
                    loss, loss_no_lamb = loss_fn(x1_outs[i],
                                                 x2_outs[i],
                                                 all_affine2_to_1=all_affine2_to_1,
                                                 all_mask_img1=all_mask_img1,
                                                 lamb=lamb,
                                                 half_T_side_dense=config.half_T_side_dense,
                                                 half_T_side_sparse_min=config.half_T_side_sparse_min,
                                                 half_T_side_sparse_max=config.half_T_side_sparse_max)

                    if avg_loss_batch is None:
                        avg_loss_batch = loss
                        avg_loss_no_lamb_batch = loss_no_lamb
                    else:
                        avg_loss_batch += loss
                        avg_loss_no_lamb_batch += loss_no_lamb

                avg_loss_batch /= config.num_sub_heads
                avg_loss_no_lamb_batch /= config.num_sub_heads

                if ((b_i % 100) == 0) or (e_i == next_epoch):
                    #writer.add_scalar('head_{}/avg_loss_batch'.format(head), avg_loss_batch.item(), b_i)
                    print(
                        "Model ind %d epoch %d head %s batch: %d avg loss %f avg loss no "
                        "lamb %f "
                        "time %s" % \
                        (config.model_ind, e_i, head, b_i, avg_loss_batch.item(),
                         avg_loss_no_lamb_batch.item(), datetime.now()))
                    sys.stdout.flush()

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

            epoch_loss.append(avg_loss)
            epoch_loss_no_lamb.append(avg_loss_no_lamb)

        # Eval
        # -----------------------------------------------------------------------

        is_best = segmentation_eval(config, net,
                                    mapping_assignment_dataloader=mapping_assignment_dataloader,
                                    mapping_test_dataloader=mapping_test_dataloader,
                                    sobel=(
                                        not config.no_sobel),
                                    using_IR=config.using_IR)

        print(
            "Pre: time %s: \n %s" % (datetime.now(), nice(config.epoch_stats[-1])))
        sys.stdout.flush()

        axarr[0].clear()
        axarr[0].plot(config.epoch_acc)
        axarr[0].set_title("acc (best), top: %f" % max(config.epoch_acc))

        axarr[1].clear()
        axarr[1].plot(config.epoch_avg_subhead_acc)
        axarr[1].set_title("acc (avg), top: %f" % max(config.epoch_avg_subhead_acc))

        axarr[2].clear()
        axarr[2].plot(config.epoch_loss_head_A)
        axarr[2].set_title("Loss head A")

        axarr[3].clear()
        axarr[3].plot(config.epoch_loss_no_lamb_head_A)
        axarr[3].set_title("Loss no lamb head A")

        axarr[4].clear()
        axarr[4].plot(config.epoch_loss_head_B)
        axarr[4].set_title("Loss head B")

        axarr[5].clear()
        axarr[5].plot(config.epoch_loss_no_lamb_head_B)
        axarr[5].set_title("Loss no lamb head B")

        fig.canvas.draw_idle()
        fig.savefig(os.path.join(config.out_dir, "plots.png"))

        if is_best or (e_i % config.save_freq == 0):
            net.module.cpu()
            save_dict = {"net": net.module.state_dict(),
                         "optimiser": optimiser.state_dict()}

            if e_i % config.save_freq == 0:
                torch.save(save_dict, os.path.join(config.out_dir, "latest.pytorch"))
                config.last_epoch = e_i  # for last saved version

            if is_best:
                torch.save(save_dict, os.path.join(config.out_dir, "best.pytorch"))

                with open(os.path.join(config.out_dir, "best_config.pickle"),
                          'wb') as outfile:
                    pickle.dump(config, outfile)

                with open(os.path.join(config.out_dir, "best_config.txt"),
                          "w") as text_file:
                    text_file.write("%s" % config)

            net.module.cuda()

        with open(os.path.join(config.out_dir, "config.pickle"), 'wb') as outfile:
            pickle.dump(config, outfile)

        with open(os.path.join(config.out_dir, "config.txt"), "w") as text_file:
            text_file.write("%s" % config)

        if config.test_code:
            print("test code completed")
            exit(0)


train()
