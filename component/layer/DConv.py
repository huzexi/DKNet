import numpy as np
import torch.nn as nn

from component.layer.convert import mode_cvt, MODE_O, MODE_S, MODE_A

__all__ = ['DConv', 'SpatialConv', 'AngularConv']


class DConv(nn.Module):
    dic = {
        'u': 2,
        'v': 3,
        'w': 4,
        'h': 5,
    }

    def __init__(self,
                 in_channels, out_channels, connection,
                 kernel_size=(3, 3),
                 n=1,
                 act=nn.ReLU,
                 stride=(1, 1),
                 padding=(1, 1)):
        super(DConv, self).__init__()
        assert len(connection) == 2, "The number of dimensions convolved should be 2."
        self.chns_in = in_channels
        self.chns_out = out_channels
        self.connection = connection
        self.body = []
        for i in range(n):
            self.body.append(nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding))
        self.body = nn.Sequential(*self.body)
        self.act = act()

    def forward(self, x):
        x = mode_cvt(x, MODE_O)
        status = x.status.copy()

        b, chns = x.size(0), x.size(1)
        sz = np.array(x.size())
        dims_1 = [self.dic[i] for i in self.connection]
        dims_0 = [self.dic[i] for i in self.dic.keys() if i not in self.connection]

        dims_map = (0, *dims_0, 1, *dims_1)
        x = x.permute(*dims_map)
        x = x.reshape(b * sz[dims_0[0]] * sz[dims_0[1]], chns, sz[dims_1[0]], sz[dims_1[1]])
        x = self.body(x)
        x = self.act(x)

        # Update size
        chns = x.size(1)
        sz[dims_1[0]], sz[dims_1[1]] = x.size(2), x.size(3)

        # Swap dims
        dims_map = dict((v, k) for k, v in enumerate(dims_map))
        dims_map = [dims_map[i] for i in range(6)]

        x = x.reshape(b, sz[dims_0[0]], sz[dims_0[1]], chns, sz[dims_1[0]], sz[dims_1[1]])
        x = x.permute(*dims_map)

        status['sz_a'] = x.size(2), x.size(3)
        status['sz_s'] = x.size(4), x.size(5)
        x.status = status

        return x


class AngularConv(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size=(3, 3),
                 n=1,
                 act=nn.ReLU,
                 stride=(1, 1),
                 padding=(1, 1)):
        super(AngularConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.body = []
        for i in range(n):
            self.body.append(nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding))
        self.body = nn.Sequential(*self.body)
        self.act = act()

    def forward(self, x):
        x = mode_cvt(x, MODE_A)
        status = x.status.copy()
        x = self.body(x)
        x = self.act(x)
        status['sz_a'] = x.size(-2), x.size(-1)
        x.status = status

        return x


class SpatialConv(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size=(3, 3),
                 n=1,
                 act=nn.ReLU,
                 stride=(1, 1),
                 padding=(1, 1)):
        super(SpatialConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.body = []
        for i in range(n):
            self.body.append(nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding))
        self.body = nn.Sequential(*self.body)
        self.act = act()

    def forward(self, x):
        x = mode_cvt(x, MODE_S)
        status = x.status.copy()
        x = self.body(x)
        x = self.act(x)
        status['sz_s'] = x.size(-2), x.size(-1)
        x.status = status

        return x
