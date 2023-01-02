#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from multiprocessing import Process, Queue, Manager
from multiprocessing.pool import Pool
from time import sleep

import cv2
import math
import numpy as np
from tqdm import tqdm
import torch
from math import log10
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from component.util import shave_border, im2double, sai_io_aid, angular_resize, lf_mod_crop, flatten_a, get_dir, \
    full2scene_name
from component.util.render import render_png

__all__ = ['infer_slices', 'net_infer', 'Evaluator']


def infer_slices(batch_x, model, scale, n_slice, offset):
    """
    Infer a LF by slices for the constraint of memory.
    :param batch_x: Input LF, (b, chn, a, a, s1, s1)
    :param model: The model to infer.
    :param scale: The SR scale.
    :param n_slice: Number of slices.
    :param offset: The overlapped length between two slices.
    :return: Output LF, (b, chn, a, a, s2, s2)
    """
    b, chn, sz_a, sz_s = batch_x.shape[0], batch_x.shape[1:2], batch_x.shape[2:4], batch_x.shape[4:6]

    slice_len = math.ceil(sz_s[0] / n_slice)
    loc_ori = list(range(0, sz_s[0], slice_len))  # loc => location
    y = torch.zeros((b,) + chn + sz_a + (sz_s[0] * scale, sz_s[1] * scale), dtype=batch_x.dtype)
    for i in range(len(loc_ori)):
        is_first = i == 0
        is_last = i == len(loc_ori) - 1

        range_ori1 = [loc_ori[i], loc_ori[i] + slice_len]
        if is_last:
            range_ori1[1] = sz_s[0]
        range_crop1 = [0, 0]
        range_crop1[0] = range_ori1[0] - offset if not is_first else range_ori1[0]
        range_crop1[1] = range_ori1[1] + offset if not is_last else range_ori1[1]
        x_slice = batch_x[:, :, :, :, range_crop1[0]:range_crop1[1], :]
        with torch.no_grad():
            y_slice = model(x_slice)

        range_ori2 = [range_ori1[0] * scale, range_ori1[1] * scale]
        range_crop2 = [range_crop1[0] * scale, range_crop1[1] * scale]
        range_crop2[0] = offset * scale if not is_first else 0
        range_crop2[1] = - offset * scale if not is_last else None
        y[:, :, :, :, range_ori2[0]:range_ori2[1], :] = y_slice[:, :, :, :, range_crop2[0]:range_crop2[1], :]
    return y


def net_infer(batch_x, model, slice_cfg=None):
    batch_x = batch_x.to(next(model.parameters()).device)
    if slice_cfg:
        slice_n, slice_offset, scale = slice_cfg['slice_n'], slice_cfg['slice_offset'], slice_cfg['scale']
        batch_pred = infer_slices(batch_x=batch_x, model=model,
                                  scale=scale, n_slice=slice_n, offset=slice_offset)
    else:
        with torch.no_grad():
            batch_pred = model(batch_x)
    batch_pred = postprocess(batch_pred)
    return batch_pred


class Evaluator:
    def __init__(self, loader, model=None, shave=None, criterion=None, mp=1, slice_cfg=None,
                 to_monitor=True, render_bgr=None):
        self.loader = loader
        self.model = model
        self.shave = shave
        self.criterion = criterion
        self.mp = mp
        self.slice_cfg = slice_cfg
        self.to_monitor = to_monitor
        self.render_bgr = render_bgr

    def set_model(self, model):
        self.model = model

    def run(self):
        # Init
        res_list = [{} for _ in range(len(self.loader))]
        if self.mp > 1:
            self._mp_init()
        else:
            pbar = tqdm(total=len(self.loader), desc="Evaluating")

        # Iterate
        for ids, (batch_x, batch_gt) in enumerate(self.loader):
            assert batch_gt.size(-1) // batch_x.size(-1) == batch_gt.size(-2) // batch_x.size(-2), \
                "Scales of X and Y should be the same"
            scale = batch_gt.size(-1) // batch_x.size(-1)

            batch_pred = net_infer(batch_x, self.model, slice_cfg=self.slice_cfg)

            batch_pred = batch_pred.to('cpu')
            batch_gt = batch_gt.to('cpu')
            batch_pred = shave_border(batch_pred, self.shave) if self.shave else batch_pred
            batch_gt = lf_mod_crop(batch_gt, scale)
            batch_gt = shave_border(batch_gt, self.shave) if self.shave else batch_gt

            if self.criterion:
                res_list[ids]['criterion'] = self.criterion(batch_pred, batch_gt)
            batch_gt, batch_pred = batch_gt.numpy(), batch_pred.numpy()

            # Calculate PSNR/SSIM
            if self.mp > 1:
                self.queue_task.put((ids, batch_pred[0], batch_gt[0]))
            else:
                res_list[ids]['psnr'], res_list[ids]['ssim'] = calc_score_lf(batch_pred[0], batch_gt[0], mp=6)
                pbar.update(len(batch_x))

            # Render BGR
            if self.render_bgr:
                render_png(batch_pred[0],
                           get_dir(self.render_bgr, full2scene_name(self.loader.dataset.names[ids])))

        # End
        if self.mp > 1:
            self._mp_end()
            while self.queue_res.qsize() != 0:
                res = self.queue_res.get()
                res_list[res[0]].update({'psnr': res[1], 'ssim': res[2]})
        else:
            pbar.close()

        return res_list

    def test_once(self):
        for _, (batch_x, _) in enumerate(self.loader):
            _ = net_infer(batch_x, self.model, slice_cfg=self.slice_cfg)
            break

    def _mp_init(self):
        self.queue_task = Queue()
        self.queue_res = Queue()

        if self.to_monitor:
            self.status = Manager().dict()
            self.status['count'] = 0
            self.status['all'] = len(self.loader)
            self.mnt = Process(target=self.monitor, args=(self.status,))
            self.mnt.start()

        self.wrk_pool = [Process(target=self.worker, args=(self.queue_task, self.queue_res, self.status))
                         for _ in range(self.mp)]
        for wrk in self.wrk_pool:
            wrk.start()

    def _mp_end(self):
        self.queue_task.put(None)
        for wrk in self.wrk_pool:
            wrk.join()

        if self.to_monitor:
            self.status['count'] = None
            self.mnt.join()

    @staticmethod
    def worker(q_task, q_res, sts):
        while True:
            msg = q_task.get()
            if msg is None:
                break
            i, gt, pred = msg
            q_res.put([i] + calc_score_lf(pred, gt))
            sts['count'] += 1
        q_task.put(None)

    @staticmethod
    def monitor(sts):
        pbar = tqdm(total=sts['all'], desc="Evaluating (MP)")
        while True:
            if sts['count'] is None:
                break
            pbar.update(sts['count'])
            sts['count'] = 0
            sleep(1)
        pbar.close()


def calc_score_lf(pred, gt, mp=1):
    # Numpy
    gt = flatten_a(gt).swapaxes(0, 1)
    pred = flatten_a(pred).swapaxes(0, 1)

    if mp > 1:
        with Pool(mp) as pool:
            res_lst = pool.starmap(calc_score_sai, zip(gt, pred))
    else:
        res_lst = [calc_score_sai(gt_it, y_it) for gt_it, y_it in zip(gt, pred)]

    return list(np.array(res_lst).mean(0))


def diff_map(img1, img2):
    diff = (abs((im2double(img1) - im2double(img2))).sum(axis=2) / 3 * 3000).astype("uint8")
    diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    return diff_color


def mse2psnr(mse):
    return 10 * log10(1 / mse.item())


def y_upsample(y, x_ycrcb, sz_a_in, sz_a_out):
    """
    Combine output Y channel with upsampled CrCb to get upsampled YCrCb.
    :param y: Output Y channel, (chn, n, s, s).
    :param x_ycrcb: Input YCrCb, (chn, a1, a1, s, s).
    :param sz_a_in: Input angular size, (a1, a1).
    :param sz_a_out: Upsampled angular size, (a2, a2), a1^2 + n = a2^2.
    :return: Upsampled YCrCb, (chn, a2, a2, s, s)
    """
    # Get angular ID
    aid_in, aid_out = sai_io_aid(sz_a_in, sz_a_out)

    # Upsampling
    lf_ycrcb = angular_resize(x_ycrcb, sz_a_out)

    # Fit output Y channel into upsampled LF
    s_sz = (lf_ycrcb.shape[3], lf_ycrcb.shape[4])
    chn = lf_ycrcb.shape[0]
    lf_ycrcb = lf_ycrcb.reshape([chn, sz_a_out[0] * sz_a_out[1], s_sz[0], s_sz[1]])
    y_ycrcb = lf_ycrcb[:, aid_out]
    y_ycrcb[0] = y

    return y_ycrcb


def calc_score_sai(gt, pred):
    """
    Calculate PSNR/SSIM score calculation.
    :param gt: Ground-truth frame, (chn, s1, s2).
    :param pred: Predicted frame, (chn, s1, s2).
    :return: PSNR score, SSIM score.
    """

    psnr_score = []
    ssim_score = []

    for chn in range(gt.shape[0]):
        psnr_score.append(psnr(gt[chn, :, :], pred[chn, :, :]))
        ssim_score.append(ssim(gt[chn, :, :], pred[chn, :, :]))

    return np.mean(psnr_score), np.mean(ssim_score)


def postprocess(y):
    """
    Process the predicted result to eliminate values more than 1 or less than 0
    :param y: Raw predicted result
    :return: Post processed result
    """
    y[y < 0] = 0
    y[y > 1] = 1
    return y
