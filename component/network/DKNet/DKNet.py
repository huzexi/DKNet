import numpy as np
import torch
import torch.nn as nn

from component.layer import SpatialConv, AngularConv, DBlock
from component.layer.convert import mode_cvt, mode_init, MODE_S, lf_cat

__all__ = ['DKNet', 'DKernel']


class DKernel(nn.Module):
    def __init__(self, chns_in, chns_out, kernel):
        super(DKernel, self).__init__()
        self.body = DBlock(
            chns_in=chns_in, chns_out=chns_out, kernel=kernel
        )

    def forward(self, x, dense_x):
        x = self.body(lf_cat([x, *dense_x], dim=1))
        return x


class DKNet(nn.Module):
    def __init__(self, config):
        super(DKNet, self).__init__()
        self.n_config = n_config = config.net_config
        self.scale = config.scale

        self.chns_feat = chns_feat = n_config.chns_feat
        self.chns_in = chns_in = 3

        self.head = [SpatialConv(in_channels=chns_in, out_channels=chns_feat)]
        self.body = []

        chns_i = chns_in if n_config.dense_i else 0
        chns_a = 0
        chns = chns_i + chns_a + chns_feat

        for i in range(n_config.n_block):
            self.body.append(DKernel(chns_in=chns, chns_out=chns_feat, kernel=n_config.kernel))
            # self.body.append(CorrBlock(chns_in=chns, chns_out=chns_feat, n_s=1))

            chns = chns_i + chns_a + chns_feat
            if n_config.dense_a:
                chns_a += chns_feat

        self.tail = [
            AngularConv(in_channels=chns,
                        out_channels=chns_feat,
                        kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            AngularConv(in_channels=chns_feat, out_channels=chns_in * config.sz_a[0] * config.sz_a[1] * config.scale ** 2,
                        kernel_size=np.array(config.sz_a)//2, padding=(0, 0)),
        ]
        self.ps = nn.PixelShuffle(config.scale)

        self.head = nn.Sequential(*self.head)
        self.body = nn.ModuleList(self.body)
        self.tail = nn.Sequential(*self.tail)

    def forward(self, inp):
        """ Input x:(batch, b, a, a, s1, s1),  Output y:(batch, b, a, a, s2, s2) """
        b, chns = inp.size(0), inp.size(1)
        inp = mode_init(inp)
        inp = mode_cvt(inp, MODE_S)
        x = inp
        status = x.status
        sz_a, sz_s = status['sz_a'], status['sz_s']

        x = self.head(x)
        dense_a = []
        dense_i = []
        if self.n_config.dense_i:
            dense_i.append(inp)
        for idx, block in enumerate(self.body):
            x = block(x, (*dense_a[:-1], *dense_i))
            x = mode_cvt(x, MODE_S)
            if self.n_config.dense_a:
                dense_a.append(x)

        x = dense_a if self.n_config.dense_a else [x]
        x = lf_cat([*x, *dense_i], dim=1)
        x = self.tail(x)
        x = x.view(b, sz_s[0], sz_s[1], chns*self.scale**2, sz_a[0]*sz_a[1])
        x = x.permute(0, 4, 3, 1, 2)
        x = x.reshape(b*sz_a[0]*sz_a[1], self.chns_in*self.scale**2, *sz_s)
        x = self.ps(x)  # (batch, c*a*a, s*n, s*n)
        x = x.view(b, *sz_a, self.chns_in, sz_s[0] * self.scale, sz_s[1] * self.scale)  # (batch, c, a, a, s*n, s*n)
        x = x.permute(0, 3, 1, 2, 4, 5)

        return x
