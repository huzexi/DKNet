from os import path

from .DefaultConfig import DefaultConfig

__all__ = ['CustomConfig']


class CustomConfig(DefaultConfig):
    dir_h5 = './data'
    dir_tmp = path.join('workdir')
    dir_tmp_train = path.join(dir_tmp, 'train')
    dir_tmp_test = path.join(dir_tmp, 'test')

    train_batch_sz = 2