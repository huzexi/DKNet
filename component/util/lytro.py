import cv2
from component.util import lf_raw2bgr, im2double, lf_bgr2ycrcb

__all__ = ['load_item']


def load_item(pth_img, a_raw, a_preserve, ycbcr=True):
    """
    Load structured BGR and YCrCb matrix from path of a raw LF.
    :param pth_img: Path of a raw LF.
    :param a_raw: Original angular size, e.g. [14, 14].
    :param a_preserve: Preserved angular size, e.g. [8, 8].
    :param ycbcr: Output YCbCr image.
    :return: Structured BGR and YCrCb matrix.
    """
    raw_lf = cv2.imread(pth_img, cv2.IMREAD_UNCHANGED)[..., :3]
    bgr_lf = lf_raw2bgr(raw_lf, a_raw, a_preserve)
    if ycbcr:
        ycrcb_lf = im2double(lf_bgr2ycrcb(bgr_lf), "float32")
        return bgr_lf, ycrcb_lf
    else:
        bgr_lf = im2double(bgr_lf, "float32")
        return bgr_lf
