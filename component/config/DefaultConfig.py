from collections import Iterable
from os import path

__all__ = ['DefaultConfig']


class DefaultConfig:
    # Directory paths
    dir_h5 = './data'
    dir_tmp = path.join('tmp')
    dir_tmp_train = path.join(dir_tmp, 'train')
    dir_tmp_test = path.join(dir_tmp, 'test')
    dir_ckp = path.join('ckp')

    # Test
    test_mp = 0
    test_slice_n = 3
    test_slice_offset = 15

    # Task properties
    dataset = 'Yeung'
    scale = 4
    shave = 15
    sz_a = (8, 8)
    color = True
    net = 'DKNet'

    # Functions
    @classmethod
    def get_train_ckp_ep(cls, ep):
        if not isinstance(cls.train_ckp_ep, Iterable):
            return cls.train_ckp_ep
        for it in cls.train_ckp_ep:
            if ep in range(it[0][0], it[0][1]):
                return it[1]
        return cls.train_ckp_ep[0][1]
