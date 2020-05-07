import glob
import os
from collections import namedtuple

from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import Cityscapes
import torch


# Based on https://github.com/mcordts/cityscapesScripts
CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                 'has_instances', 'ignore_in_eval', 'color'])

classes = [
    CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]

# map id to train_id, with 255 or -1 mapped to 19
pixLabels = {0: 19, 1: 19, 2: 19, 3: 19, 4: 19, 5: 19, 6: 19, 7: 0, 8: 1, 9: 19,
             10: 19, 11: 2, 12: 3, 13: 4, 14: 19, 15: 19, 16: 19, 17: 5, 18: 19,
             19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13,
             27: 14, 28: 15, 29: 19, 30: 19, 31: 16, 32: 17, 33: 18, -1: 19}

# print(classes[34][7][2])

# define transforms
resize_pil = transforms.Resize((256, 512))  # use 8, 16 to test
to_t = transforms.ToTensor()
to_resized_tensor = transforms.Compose([resize_pil, to_t])
# to_pil = transforms.ToPILImage()


def one_hot_transform(seg_tensor):
    seg_tensor[:, :, :] *= 255  # seg array is floats = category/255
    print(seg_tensor[0])
    # for x in seg_tensor:
    #     print(x)
    #     x = pixLabels[x]
    #     print(x)
    # print(seg_tensor[0])
    # seg_tensor[seg_tensor[:, :, :] == -1] = 34
    # assume -1 is fitered out? I think this is handled by dataloader
    height = list(seg_tensor.shape)[1]
    width = list(seg_tensor.shape)[2]
    for i in range(0, height):
        for j in range(0, width):
            seg_tensor[0][i][j] = pixLabels[int(seg_tensor[0][i][j])]  # possible rounding errors?
    # 35 classes: 0-33 and -1  # only 19 classes used for training and eval
    print(seg_tensor[0])
    ret_tensor = torch.zeros(20, height, width)
    for chan in range(0, 20):
        ret_tensor[chan, :, :] = seg_tensor[0, :, :] == chan
    # ret_tensor[34, :, :] = ((seg_tensor[0, :, :] == -1) or (seg_tensor[0, :, :] == 34))
    return ret_tensor


class CityscapesLoader(torch.utils.data.Dataset):
    # split must be 'train', 'test', or 'val'
    def __init__(self, split):
        super(CityscapesLoader, self).__init__()
        self.tensors_dataset = Cityscapes(root='./data/cityscapes', split=split, mode='fine', target_type='semantic',
                                          transform=to_resized_tensor, target_transform=to_resized_tensor)

    def __getitem__(self, item):
        img, seg = self.tensors_dataset[item]
        seg = one_hot_transform(seg)  # .squeeze(0)  # need to remove classes dimension for CrossEntropyLoss
        # seg = seg.long()
        return img, seg

    def __len__(self):
        return len(self.tensors_dataset)


# train_dataset = CityscapesLoader('train')
# for i in range(0, 1):
#     img, seg = train_dataset[i]
#     # print(img)
#     print(seg[0])



# the higher-order representation of edges captured by network is different than simple L1Loss
def get_edges(pred_seg, gt_seg):
    edge_tensor = gt_seg - pred_seg
    # a mismatched edge will have 1s in the channel that is the gt, and -1s in the channel that was predicted
    # softmax predicts one channel only to put positive 1 in
    # remove negative values only
    # edge_tensor = torch.nn.functional.relu(edge_tensor, inplace=False)  # inplace=True would be faster? default False
    # use tanh instead of softmax to predict both 1s and -1s
    return edge_tensor


# need to change models' channel dimensions to use this instead
def get_edges2(pred_seg, gt_seg):
    edge_tensor = gt_seg - pred_seg
    # remove negative values only
    missed_edge_tensor = torch.nn.functional.relu(edge_tensor)
    # flip negatives to positive, then get those values only
    edge_tensor[:, :, :] *= -1
    wrong_edge_tensor = torch.nn.functional.relu(edge_tensor)
    ret_tensor = torch.cat([missed_edge_tensor, wrong_edge_tensor], 0)
    return ret_tensor


# intersection over union assuming both tensors are [BATCH_SIZE, classes, h, w]
def iou(outputs: torch.Tensor, labels: torch.Tensor, batch_size=1):
    outputs = outputs.argmax(dim=1, keepdim=True)  # dim is classes
    height = list(outputs.shape)[2]
    width = list(outputs.shape)[3]
    # class 20 not used in eval, so copy first 19 channels to new tensor
    top_pred = torch.zeros(batch_size, 19, height, width).cuda()  # assume batch size 1
    labels_19 = torch.zeros(batch_size, 19, height, width).cuda()
    for chan in range(0, 19):
        top_pred[:, chan, :, :] = outputs[:, 0, :, :] == chan
        labels_19[:, chan, :, :] = labels[:, chan, :, :]

    labels_19 = labels_19.int()
    top_pred = top_pred.int()
    SMOOTH = 1e-6
    intersection = torch.mul(top_pred, labels_19).int().sum((2, 3))  # both 1 -> 1
    union = intersection + torch.logical_xor(top_pred, labels_19).int().sum((2, 3))  # 1 in either -> 1
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    # iou is score for every class in every batch
    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    ret = iou.mean().item()  # averages across whole batch
    return ret

# o = torch.randn([1, 20, 2, 3]).abs()
# l = torch.randn([1, 20, 2, 3]).int().abs()
# print(l.shape)
# print(iou(o, l))
# print(l.shape)


# seg is [classes, h, w]
def seg_to_rgb(seg):
    seg = seg.argmax(axis=0)  # .squeeze(1)  argmax squeezes, and squeeze doesn't change shape?
    seg = seg.squeeze(1)
    h = list(seg.shape)[0]
    w = list(seg.shape)[1]
    png = torch.zeros([3, h, w])
    for i in range(0, h):
        for j in range(0, w):
            color = classes[seg[i][j]][7]
            png[0][i][j] = color[0]
            png[1][i][j] = color[1]
            png[2][i][j] = color[2]
    return png


