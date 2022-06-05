import os
import subprocess
from os import path
import cv2
import numpy as np


def get_dir(*pth):
    """join dir pth and create it when it doesn't exist"""
    pth_ = path.join(*pth)
    if not path.exists(pth_):
        os.makedirs(pth_)
    return pth_


def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode()
    except:
        return None


def im2double(im, dtype='float64'):
    info = np.iinfo(im.dtype)
    return im.astype(dtype) / info.max


def double2im(im, dtype='uint16'):
    info = np.iinfo(dtype)
    return (im * info.max).astype(dtype)


def np2torch(mat, batch=False):
    shape = mat.shape
    dims = list(range(len(shape)))
    dims.insert(1 if batch else 0, dims[-1])
    dims.pop()
    return mat.transpose(dims)


def torch2np(mat, batch=False):
    shape = mat.shape
    dims = list(range(len(shape)))
    dims.pop(1 if batch else 0)
    dims.append(0)
    return mat.transpose(dims)


def shave_border(img, bd, chn_first=True):
    """Shave border area of spatial views. A common operation in SR."""
    if chn_first:
        return img[..., bd:-bd, bd:-bd]
    else:
        return img[bd:-bd, bd:-bd]


def sai_io_map(a_in, a_out):
    """
    Build a matrix indicating the I/O SAI location.
    :param a_in: Input SAIs' size.
    :param a_out: Output SAIs' size.
    :return: A matrix of size a_out, 1 => output, 0 => input.
    """
    sai_map = np.ones(a_out).astype(np.int)
    step = (np.array(a_out) - np.array(a_in)) // (np.array(a_in) - 1) + 1

    for y in range(0, a_out[0], step[0]):
        for x in range(0, a_out[1], step[1]):
            sai_map[y, x] = 0

    return sai_map


def sai_io_aid(a_in, a_out):
    """ Flatten angular id of sai_io_map. """
    idx_mat = np.array(range(a_out[0] * a_out[1])).reshape(a_out)
    sai_map = sai_io_map(a_in, a_out)

    aid_in = idx_mat[sai_map == 0]
    aid_out = idx_mat[sai_map == 1]
    return aid_in, aid_out


def lf_raw2bgr(raw_lf, a_sz, a_preserve):
    """
    Convert raw LF to structured BGR color matrix.
    :param raw_lf: Raw image matrix (a*n, a*n, 3).
    :param a_sz: Size of angular dims.
    :param a_preserve: Number of preserved angular views.
    :return: BGR color matrix (a, a, n, n, 3).
    """
    s_sz = (raw_lf.shape[0:2] / np.array(a_sz)).astype(int)
    bgr_lf = np.zeros([
        a_sz[0], a_sz[1],
        s_sz[0], s_sz[1], 3,
    ]).astype(raw_lf.dtype)
    for ay in range(a_sz[0]):
        for ax in range(a_sz[1]):
            bgr_lf[ay, ax, :, :, :] = raw_lf[ay::a_sz[0], ax::a_sz[1], :3]
    if a_sz != a_preserve:
        off = ((np.array(a_sz) - np.array(a_preserve)) / 2).astype(int)  # Preserve middle area and cast out margin
        bgr_lf = bgr_lf[off[0]:-off[0], off[1]:-off[1], :, :, :]  # [3:11, 3:11, 3, :, :]
    return bgr_lf


def lf_bgr2ycrcb(bgr_lf):
    """Convert structured BGR color matrix to structured YCrCb color matrix."""
    return lf_cvt(bgr_lf, cv2.COLOR_BGR2YCrCb)


def lf_cvt(src, code):
    """Convert structured matrix to other color space."""
    ycrcb_lf = np.zeros(src.shape).astype(src.dtype)
    for ay in range(src.shape[1]):
        for ax in range(src.shape[2]):
            img = src[:, ay, ax, :, :]
            img = img.transpose([1, 2, 0])
            img = cv2.cvtColor(img, code)
            ycrcb_lf[:, ay, ax, :, :] = img.transpose([2, 0, 1])
    return ycrcb_lf


def lf_mod_crop(lf, scale):
    """Crop the LF to fit the upsampled spatial size."""
    dim = len(lf.shape)
    sz_s = np.array(lf.shape[-2:]) // scale * scale
    return lf[..., :sz_s[0], :sz_s[1]]


def ycrcb2bgr(ycrcb, channel_first=False):
    """
    Convert image from YCrCb to BGR, fixing the out-of-range problem.
    :param ycrcb: YCrCb image, should be [0, 1].
    :return: BGR image
    """
    if channel_first:
        ycrcb = ycrcb.transpose([1, 2, 0])
    bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    bgr[bgr < 0] = 0
    bgr[bgr > 1] = 1
    if channel_first:
        bgr = bgr.transpose([2, 0, 1])
    return bgr


def angular_resize(lf, sz_a_dst):
    """
    Resize angularly.
    :param lf: Input LF, (chn, a1, a1, s, s).
    :param sz_a_dst: Destination angular size, (a2, a2).
    :return: Resized LF matrix, (chn, a2, a2, s, s).
    """
    a_sz = (lf.shape[1], lf.shape[2])
    s_sz = (lf.shape[3], lf.shape[4])
    chn = lf.shape[0]
    lf = np.transpose(lf, [3, 4, 1, 2, 0]).reshape([s_sz[0] * s_sz[1], a_sz[0], a_sz[1], chn])

    lf_up = []
    for i in range(lf.shape[0]):
        # TODO: Test which interpolation is better
        up = cv2.resize(lf[i], dsize=(sz_a_dst[0], sz_a_dst[1]), interpolation=cv2.INTER_LINEAR)
        lf_up.append(up)
    lf_up = np.array(lf_up)
    lf_up = np.transpose(lf_up, [3, 1, 2, 0]).reshape([chn, sz_a_dst[0], sz_a_dst[1], s_sz[0], s_sz[1]])
    return lf_up


def spatial_resize(lf, sz_s_dst):
    """
    Resize spatially.
    :param lf: LF matrix (a, a, s1, s1, chn)
    :param sz_s_dst: Destination spatial size (s2, s2)
    :return: Resized LF matrix (a, a, s2, s2, chn)
    """
    a_sz = (lf.shape[0], lf.shape[1])
    s_sz = (lf.shape[2], lf.shape[3])
    chn = lf.shape[4]
    lf = lf.reshape([a_sz[0] * a_sz[1], s_sz[0], s_sz[1], chn])

    lf_up = []
    for i in range(lf.shape[0]):
        up = cv2.resize(lf[i], dsize=(sz_s_dst[1], sz_s_dst[0]), interpolation=cv2.INTER_LINEAR)
        lf_up.append(up)
    lf_up = np.array(lf_up)
    lf_up = lf_up.reshape([a_sz[0], a_sz[1], sz_s_dst[0], sz_s_dst[1], chn])
    return lf_up


def out_spatial_resize(lf, sz_s_dst):
    """
    Resize spatially of output.
    :param lf: LF matrix (a, s1, s1, chn)
    :param sz_s_dst: Destination spatial size (s2, s2)
    :return: Resized LF matrix (a, s2, s2, chn)
    """

    lf_up = []
    for i in range(lf.shape[0]):
        up = cv2.resize(lf[i], dsize=(sz_s_dst[1], sz_s_dst[0]), interpolation=cv2.INTER_LINEAR)
        lf_up.append(up)
    lf_up = np.array(lf_up)
    return lf_up


def full2scene_name(pth):
    """Extract img's name from its path."""
    return path.splitext(path.basename(pth))[0]


def transpose_out(ycrcb_up, a_in, a_out):
    """
    Transpose output for Henry's evaluation (in MATLAB).
    For example, ycrcb_up is 60 views in 8*8 output, after transposing it will fit in transposed 8*8 output too.
    """
    in_sai, out_sai = sai_io_aid(a_out, a_in)

    idx_t = np.zeros(a_out[0] * a_out[1]).astype(np.int)
    idx_t[out_sai] = range(len(out_sai))
    idx_t = idx_t.reshape(a_out).transpose((1, 0)).reshape(a_out[0] * a_out[1])
    idx_t = idx_t[out_sai]

    return ycrcb_up[idx_t]


def flatten_a(lf):
    shape = lf.shape
    sz_a = shape[-4:-2]
    sz_s = shape[-2:]
    return lf.reshape(shape[:-4] + (sz_a[0]*sz_a[1],) + sz_s)


def sync_device(src, dst):
    if hasattr(src, "parameters"):
        device = next(src.parameters()).device
    else:
        device = src.device
    return dst.to(device)
