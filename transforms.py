import cv2
import numpy as np
import torch
import torch.nn.functional as F


def perform_affine_tf(data, tf_matrices):
    # expects 4D tensor, we preserve gradients if there are any
    
    n_i, k, h, w = data.shape
    n_i2, r, c = tf_matrices.shape
    assert (n_i == n_i2)
    assert (r == 2 and c == 3)

    grid = F.affine_grid(tf_matrices, data.shape)  # output should be same size
    data_tf = F.grid_sample(data, grid,
                            padding_mode="zeros")  # this can ONLY do bilinear

    return data_tf

def random_translation_multiple(data, half_side_min, half_side_max):
    n, c, h, w = data.shape
        
    # pad last 2, i.e. spatial, dimensions, equally in all directions
    data = F.pad(data,
                 (half_side_max, half_side_max, half_side_max, half_side_max), "constant", 0)
    assert (data.shape[2:] == (2 * half_side_max + h, 2 * half_side_max + w))

    # random x, y displacement
    t = np.random.randint(half_side_min, half_side_max + 1, size=(2,))
    polarities = np.random.choice([-1, 1], size=(2,), replace=True)
    t *= polarities

    # -x, -y in orig img frame is now -x+half_side_max, -y+half_side_max in new
    t += half_side_max

    data = data[:, :, t[1]:(t[1] + h), t[0]:(t[0] + w)]
    assert (data.shape[2:] == (h, w))

    return data

