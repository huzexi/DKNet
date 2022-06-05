from .BaseDataset import BaseDataset
from .YeungDataset import YeungDataset

__all__ = ['get_dataset']


def get_dataset(name, config, mode=BaseDataset.MODE_TRAIN):
    dic = {
        'Yeung': YeungDataset,
    }
    if name not in dic.keys():
        raise ImportError("Unknown dataset.")
    return dic[name](config, mode)
