from .DefaultConfig import DefaultConfig

__all__ = ['CustomConfig']


class CustomConfig(DefaultConfig):
    dir_h5 = './data'
