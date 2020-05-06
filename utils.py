import torch
from random import randrange


# img is [batch_size, 3, h, w] for batch of RGB images
def remove_catx(img, seg, category_idx):
    pred_idxs = torch.argmax(seg, dim=1, keepdim=True)
    # mask = torch.ones(img.squeeze(dim=1).size)  # size of batch with one channel per image
    mask = pred_idxs != category_idx  # zeroes where catx predicted
    blacked = img * mask  # all 3 color channels set to 0 at catx indices
    return blacked


# img is [batch_size, 3, h, w] for batch of RGB images
def remove_random_region(img):
    # 2 regions 1/32th to 1/8th of image, horizontal and vertical rectangles
    # h = list(img.size())[2]
    # w = list(img.size())[3]
    batch_size, k, h, w = img.shape
    region_h = h // 8
    region_w = w // 8
    # print("h, w, rh, rw", h, w, region_h, region_w)
    # define height and width of 2 patches
    dims1 = [region_h * randrange(1, 2), region_w * randrange(2, 4)]
    # print('dims1', dims1)
    dims2 = [region_h * randrange(2, 4), region_w * randrange(1, 2)]
    # print('dims2', dims2)
    # place patch top left corner
    top_left_1 = list((randrange(0, h - dims1[0]), randrange(0, w - dims1[1])))
    # print(top_left_1)
    # define coordinates of patch as top left and bottom right corners (h1, w1, h2, w2)
    coords1 = top_left_1
    coords1.extend([top_left_1[0] + dims1[0], top_left_1[1] + dims1[1]])
    # print(coords1)
    top_left_2 = list((randrange(0, h - dims2[0]), randrange(0, w - dims2[1])))
    # print(top_left_2)
    coords2 = top_left_2
    coords2.extend([top_left_2[0] + dims2[0], top_left_2[1] + dims2[1]])
    # print(coords2)
    mask = torch.ones([h, w])

    for i in range(0, h):
        for j in range(0, w):
            if (coords1[0] < i < coords1[2]) and (coords1[1] < j < coords1[3]):
                mask[i][j] = 0
            if (coords2[0] < i < coords2[2]) and (coords2[1] < j < coords2[3]):
                mask[i][j] = 0

    mask = mask.view(1, 1, h, w)
    mask = mask.repeat(batch_size, 1, 1, 1)

    img_b = img * mask

    return img_b


def custom_loss_g():
    return 0


def custom_loss_iic():
    return 0


# test_img = torch.rand([2, 3, 16, 20])
# print(test_img)
# # seg = torch.rand([2, 6, 4, 5])  # if there were 6 categories
# # ret = remove_catx(test_img, seg, 4)
# ret = remove_random_region(test_img)
# print(ret)

