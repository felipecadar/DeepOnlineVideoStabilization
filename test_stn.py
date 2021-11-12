from numpy.core.fromnumeric import shape
from numpy.lib import npyio
from numpy.lib.shape_base import apply_along_axis
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as  np
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    original_img = cv2.imread('img.jpeg')
    norm_img = original_img.astype(np.float64) / 255

    x = torch.tensor(norm_img)
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
    else:
        x = x.unsqueeze(0).permute(0,3,1,2)


    n_side = 5
    nx, ny = (n_side, n_side)

    xv, yv = np.meshgrid(np.linspace(-1, 1, nx),  np.linspace(-1, 1, ny))
    np_grid = np.stack([xv, yv])
    np_grid = np_grid + (np.random.sample(np_grid.shape) /10)

    # np_grid[0, 1, 1] = 0.5
    # np_grid[1, 1, 1] = 0.5

    # print(np_grid[0,...])
    # print(np_grid[1,...])
    
    grid = torch.tensor(np_grid)
    grid = grid.unsqueeze(0)
    grid = F.interpolate(grid, size=[x.shape[2], x.shape[3]], align_corners=True, mode='bilinear').permute(0, 2, 3, 1)

    print('Grid Shape: {}'.format(grid.shape))
    print('X    Shape: {}'.format(x.shape))
    new_x = F.grid_sample(x, grid, align_corners=True)
    print('New X Shape: {}'.format(new_x.shape))
    
    new_img = (new_x.squeeze(0).permute(1,2,0).numpy() * 255).astype(np.uint8).copy()
    print('New Img Shape: {}'.format(new_img.shape))

    h,w = original_img.shape[:2]
    for i in range(n_side):
        for j in range(n_side):
            pos_x = int(np.interp(np_grid[0, i, j], [-1, 1], [0, w-1]))
            pos_y = int(np.interp(np_grid[1, i, j], [-1, 1], [0, h-1]))

            new_img = cv2.circle(new_img, [pos_x, pos_y], 4, [0,0,255], -1)
            
            # pt = grid[0, pos_x, pos_y].numpy()
            # pt = np.interp(pt, [-1, 1], [0, w-1]).astype(int)
            # new_img = cv2.circle(new_img, [pt[0], pt[1]], 4, [0,255,0], -1)
            # print(pt)


    # grid = np.interp(grid[0].numpy(), [-1, 1], [0, w-1])
    # plt.scatter(grid[:,:,0].ravel(), grid[:,:,1].ravel(), alpha=0.5)
    # plt.show()

    cv2.imshow('1', original_img)
    cv2.imshow('2', new_img)
    cv2.waitKey(0)

