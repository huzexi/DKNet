from abc import abstractmethod
from os import path
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from component.util import sai_io_aid
from component.util.augment import flip, rotate


class BaseDataset(Dataset):
    MODE_TRAIN = 1
    MODE_TEST = 2
    MODE_VALID = 3
    MODE_PREPARE = -1

    path_h5 = {
        MODE_TRAIN: '',
        MODE_TEST: ''
    }

    sz_a = (-1, -1)
    sz_s = (-1, -1)  # Spatial dimensions will be resized to this size when necessary, e.g. testing when training.

    def __init__(self, config, mode=MODE_TRAIN):
        self.config = config
        self.mode = mode
        self.scale = config.scale
        self.color = config.color
        
        if mode == self.MODE_TRAIN:
            self.augment_config = config.train_aug
            self.patch_sz = config.train_patch_sz

        if mode != self.MODE_PREPARE:
            with h5py.File(self.get_h5_path(self.mode), 'r') as h5:
                self.names = self.get_names(h5)
                self.n_scenes = len(self.names) if mode != self.MODE_PREPARE else None

    def __len__(self):
        if self.mode == self.MODE_TRAIN:
            return self.config.train_ep_iter * self.config.train_batch_sz
        elif self.mode in [self.MODE_TEST, self.MODE_VALID]:
            return self.n_scenes

    def __getitem__(self, idx):
        with h5py.File(self.get_h5_path(self.mode), 'r') as h5:
            if self.mode == self.MODE_TRAIN:
                idx = torch.randint(0, self.n_scenes, (1,))
                name = self.names[idx]
                if self.color:
                    hr = h5[name + '/rgb/original']
                    lr = h5[name + '/rgb/%dx' % self.scale]
                    [hr, lr] = self.process(hr, lr, patch_sz=self.patch_sz, chns=np.arange(0, 3))
                else:
                    hr = h5[name + '/ycbcr/original']
                    lr = h5[name + '/ycbcr/%dx' % self.scale]
                    [hr, lr] = self.process(hr, lr, patch_sz=self.patch_sz, chns=np.arange(0, 1))
                [hr, lr] = self.data_augment([hr, lr], self.augment_config)
            elif self.mode in [self.MODE_TEST, self.MODE_VALID]:
                name = self.names[idx]
                if self.color:
                    hr = h5[name + '/rgb/original']
                    lr = h5[name + '/rgb/%dx' % self.scale]
                    [hr, lr] = self.process(hr, lr, patch_sz=None, chns=np.arange(0, 3))
                else:
                    hr = h5[name + '/ycbcr/original']
                    lr = h5[name + '/ycbcr/%dx' % self.scale]
                    chns = np.arange(0, 1) if self.mode == self.MODE_VALID else np.arange(0, 3)
                    [hr, lr] = self.process(hr, lr, patch_sz=None, chns=chns)
        return lr.copy(), hr.copy()

    def get_h5_path(self, mode):
        if mode in [self.MODE_TRAIN]:
            mode = self.MODE_TRAIN
        elif mode in [self.MODE_VALID, self.MODE_TEST]:
            mode = self.MODE_TEST
        return path.join(self.config.dir_h5, self.path_h5[mode])

    def get_names(self, h5=None):
        return h5.attrs['names'].split(',')

    @abstractmethod
    def prepare(self):
        raise NotImplementedError

    def before_process(self, hr, lr):
        return hr, lr

    def process(self, hr, lr, patch_sz, chns=np.arange(0, 1)):
        hr, lr = self.before_process(hr, lr)
        if patch_sz is not None:
            # Crop
            sz_s = lr.shape[-2:]
            sx = random.randint(0, sz_s[1] - patch_sz[1])
            sy = random.randint(0, sz_s[0] - patch_sz[0])
            ratio = np.array(hr.shape[-2:]) // lr.shape[-2:]
            region = np.array([
                [sy, sy + patch_sz[0]],
                [sx, sx + patch_sz[1]]
            ])
            lr = lr[chns, ..., region[0][0]:region[0][1], region[1][0]:region[1][1]]

            region = (region * ratio).astype(int)
            hr = hr[chns, ..., region[0][0]:region[0][1], region[1][0]:region[1][1]]
        else:
            hr = hr[chns]
            lr = lr[chns]
        hr, lr = self.after_process(hr, lr)
        return hr, lr

    def after_process(self, hr, lr):
        return hr, lr

    @classmethod
    def data_augment(cls, lfs, aug_config):
        to = torch.randint(0, 2, (2,))
        to_flip = aug_config['flip'] and to[0] == 1
        to_rotate = aug_config['rotate'] and to[1] == 1
        lfs = flip(lfs) if to_flip else lfs
        lfs = rotate(lfs) if to_rotate else lfs

        return lfs

    @classmethod
    def get_xy(cls, lf, a_in):
        """
        LF to (x, y).
        :param lf: LF data, (v, u, y, x, c).
        :param a_in: Input angular size.
        :return: (x, y) => ((n, y, x, c), (u', v', y, x, c)).
        """
        sz_a = (lf.shape[0], lf.shape[1])
        sz_s = (lf.shape[2], lf.shape[3])
        sz_c = (lf.shape[4])
        aid_in, aid_out = sai_io_aid(a_in, lf.shape[0:2])
        lf = lf.reshape([sz_a[0] * sz_a[1], sz_s[0], sz_s[1], sz_c])
        lf_in, lf_out = lf[aid_in], lf[aid_out]
        lf_in = lf_in.reshape([a_in[0], a_in[1], sz_s[0], sz_s[1], sz_c])
        return lf_in, lf_out
