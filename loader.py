import torchvision
import torchvision.transforms as transforms
import torch

# scripts
from image_loader import *
from net10a_twohead import *
from IIC_Losses import *


h, w, in_channels = 240, 240, 3

# Lists to keep track of progress
img_list = []
G_losses = []  # maybe make sure just shadow mask prediction is working first before we test GAN
D_losses = []
iters = 0
lamb = 1.0
batch_sz = 8
num_sub_heads = 1
half_T_side_dense = 0
half_T_side_sparse_min = 0
half_T_side_sparse_max = 0

# Defining the learning rate, number of epochs and beta for the optimizers
lr = 0.001
beta1 = 0.5
num_epochs = 10

# Create the models
net = SegmentationNet10a(num_sub_heads)

# Initialize IIC objective function
loss_fn = IID_segmentation_loss

# Setup Adam optimizers for both
optimiser = torch.optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.1))

transform = transforms.Compose([transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()])
# Creates a dataloader for the model
dataloader = DataLoader(dataset=Data(h, w, transform),  # what data does this load? we just want to load images with shadows, right?
        batch_size=batch_sz, shuffle=True, drop_last=True)  # shuffle is to pick random images and drop last is to drop the last batch so the size does not changes


for e_i in range(0, num_epochs):
    print("Starting e_i: %d " % (e_i))

    avg_loss = 0.  # over heads and head_epochs (and sub_heads)
    avg_loss_no_lamb = 0.
    avg_loss_count = 0

    for data in dataloader:
        img1, img2, affine2_to_1, mask_img1 = data

        net.zero_grad()

        pre_channels = in_channels

        x1_outs = net(img1)
        x2_outs = net(img2)

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
        avg_loss_no_lamb_batch /= num_sub_heads

        # check the print statement that was here
        print("e {} al {} aln {}".format(e_i, avg_loss_batch.item(), avg_loss_no_lamb_batch.item()))

        if not np.isfinite(avg_loss_batch.item()):
            print("Loss is not finite... %s:" % str(avg_loss_batch))
            exit(1)

        avg_loss += avg_loss_batch.item()
        avg_loss_no_lamb += avg_loss_no_lamb_batch.item()
        avg_loss_count += 1

        avg_loss_batch.backward()
        optimiser.step()

        # this is to test that the output makes sense
        o = transforms.ToPILImage()(img1[0])
        o.save("test1.png")
        o = transforms.ToPILImage()(img2[0])
        o.save("test2.png")
        im = torch.argmax(x1_outs[0], dim=1).detach().numpy()
        cv2.imwrite('test.png', im[0] * 255)

        torch.cuda.empty_cache()


    avg_loss = float(avg_loss / avg_loss_count)
    avg_loss_no_lamb = float(avg_loss_no_lamb / avg_loss_count)
