import torch.nn as nn

from . import DConv

__all__ = ['DBlock']


class DBlock(nn.Module):
    def __init__(self, chns_in, chns_out, kernel):
        super().__init__()
        dict_kernel = {
            'SAS':      ['wh', 'uv'],
            'Alpha':    ['wh', 'uv', 'vh', 'uw'],
            'Beta':     ['wh', 'uv', 'vw', 'uh'],
            'Gamma':    ['wh', 'uv', 'vw', 'vh', 'uw', 'uh'],
            'EPI1':     ['uw', 'vh'],
            'EPI2':     ['vw', 'uh'],
            'EPI3':     ['uw', 'vh', 'vw', 'uh'],
            'Dup1-4':   ['wh', 'wh', 'wh', 'uv'],
            'Dup2-4':   ['wh', 'uv', 'uv', 'uv'],
            'Dup1-6':   ['wh', 'wh', 'wh', 'wh', 'wh', 'uv'],
            'Dup2-6':   ['wh', 'uv', 'uv', 'uv', 'uv', 'uv'],
        }
        self.body = []
        for i, conn in enumerate(dict_kernel[kernel]):
            self.body.append(
                DConv(in_channels=chns_in if i == 0 else chns_out, out_channels=chns_out, connection=conn),
            )
        self.body = nn.Sequential(*self.body)

    def forward(self, x):
        x = self.body(x)
        return x
