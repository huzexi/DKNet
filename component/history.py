import os
from os import path


class History:
    def __init__(self, pth, attrs):
        self.pth = pth
        self.attrs = attrs

    def init(self):
        if path.exists(self.pth):
            os.remove(self.pth)
        with open(self.pth, 'w') as f:
            f.write(','.join(['epoch', *self.attrs]) + '\n')

    def write(self, ep, vals):
        with open(self.pth, 'a') as f:
            f.write(','.join([str(ele) for ele in [ep, *vals]]) + '\n')
