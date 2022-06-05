from os import path
import cv2

from component.util import double2im

__all__ = ['render_png']


def render_png(lf, pth):
    lf = double2im(lf)
    for idv in range(lf.shape[-4]):
        for idu in range(lf.shape[-3]):
            cv2.imwrite(path.join(pth, "%02d-%02d.png") % (idv, idu), lf[:, idv, idu].transpose((1, 2, 0)))
