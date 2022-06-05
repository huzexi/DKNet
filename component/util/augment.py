import numpy as np
import torch

__all__ = ['flip', 'rotate']


def flip(lfs):
    mode = torch.randint(0, 1+1, (1,))
    dst = []
    if mode == 0:
        for lf in lfs:
            lf = np.flip(lf, 1)
            lf = np.flip(lf, 3)
            dst.append(lf)
    elif mode == 1:
        for lf in lfs:
            lf = np.flip(lf, 2)
            lf = np.flip(lf, 4)
            dst.append(lf)
    return dst


def rotate(lfs):
    angle = torch.randint(1, 3+1, (1,))
    dst = []
    for lf in lfs:
        lf = np.rot90(lf, angle, (1, 2))
        lf = np.rot90(lf, angle, (3, 4))
        dst.append(lf)
    return dst
