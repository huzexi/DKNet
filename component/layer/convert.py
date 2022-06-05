import torch

MODE_O = 0
MODE_A = 1
MODE_S = 2


def convert(x, src, dst, sz=()):
    if src == MODE_O:
        b, chns, sz_a, sz_s = x.size(0), x.size(1), (x.size(2), x.size(3)), (x.size(4), x.size(5))
        if dst == MODE_S:
            x = x.permute(0, 2, 3, 1, 4, 5)     # (batch, a, a, c, s, s)
            x = x.reshape(-1, chns, *sz_s)      # (batch*a*a, c, s, s)
    elif src == MODE_A and dst == MODE_S or src == MODE_S and dst == MODE_A:
        sz1 = sz
        chns, sz2 = x.size(1), (x.size(2), x.size(3))
        x = x.reshape(-1, *sz1, chns, *sz2)     # (batch, d1, d1, c, d2, d2)
        x = x.permute(0, 4, 5, 3, 1, 2)         # (batch, d2, d2, c, d1, d1)
        x = x.reshape(-1, chns, *sz1)           # (batch*s*s, c, a, a)
    else:
        raise NotImplementedError
    return x


def mode_init(x):
    x.status = {
        'sz_a': (x.size(2), x.size(3)),
        'sz_s': (x.size(4), x.size(5)),
        'mode': MODE_O
    }
    return x


def mode_cvt(x, dst):
    status = x.status.copy()
    src = status['mode']
    if src == dst:
        pass
    elif src == MODE_O:
        b, chns, sz_a, sz_s = x.size(0), x.size(1), status['sz_a'], status['sz_s']
        status['sz_a'], status['sz_s'] = sz_a, sz_s
        if dst == MODE_S:
            x = x.permute(0, 2, 3, 1, 4, 5)     # (batch, a, a, c, s, s)
            x = x.reshape(b*sz_a[0]*sz_a[1], chns, *sz_s)      # (batch*a*a, c, s, s)
        elif dst == MODE_A:
            x = x.permute(0, 4, 5, 1, 2, 3)  # (batch, s, s, c, a, a)
            x = x.reshape(b * sz_s[0] * sz_s[1], chns, *sz_a)  # (batch*s*s, c, a, a)
    elif (src, dst) in ((MODE_A, MODE_S), (MODE_S, MODE_A)):
        (sz1, sz2) = (status['sz_a'], status['sz_s']) if src == MODE_S else (status['sz_s'], status['sz_a'])
        b = x.size(0) // (sz1[0] * sz1[1])
        chns = x.size(1)
        x = x.reshape(b, *sz1, chns, *sz2)      # (batch, d1, d1, c, d2, d2)
        x = x.permute(0, 4, 5, 3, 1, 2)         # (batch, d2, d2, c, d1, d1)
        x = x.reshape(b*sz2[0]*sz2[1], chns, *sz1)            # (batch*s*s, c, a, a)
        (status['sz_a'], status['sz_s']) = (sz1, sz2) if src == MODE_S else (sz2, sz1)
    elif src in (MODE_A, MODE_S) and dst == MODE_O:
        (sz1, sz2) = (status['sz_a'], status['sz_s']) if src == MODE_S else (status['sz_s'], status['sz_a'])
        b = x.size(0) // (sz1[0] * sz1[1])
        chns = x.size(1)
        x = x.reshape(b, *sz1, chns, *sz2)      # (batch, d1, d1, c, d2, d2)
        if src == MODE_S:
            x = x.permute(0, 3, 1, 2, 4, 5)
        else:
            x = x.permute(0, 3, 4, 5, 1, 2)
        (status['sz_a'], status['sz_s']) = (sz1, sz2) if src == MODE_S else (sz2, sz1)
    else:
        raise NotImplementedError
    status['mode'] = dst
    x.status = status
    return x


def lf_cat(tensors, dim):
    status = tensors[0].status.copy()
    tensors = torch.cat(tensors, dim)
    tensors.status = status
    return tensors
