from os import path
import h5py
import numpy as np

from . import BaseDataset
from component import logger
from component.util import spatial_resize
from component.util.lytro import load_item

__all__ = ['LytroDataset']


class LytroDataset(BaseDataset):
    # Configuration
    path_h5 = {
        BaseDataset.MODE_TRAIN: 'Lytro_Train.h5',
        BaseDataset.MODE_TEST: 'Lytro_Test.h5'
    }

    source_dir = {
        # Path to the folder containing the folders of general, occlusions, reflective, etc.
        'Stanford': '/path/to/Stanford/',
        # Path to the folder containing TrainingSet and TestSet
        'SIGASIA16': '/path/to/SIGASIA16/'
    }

    sz_a_raw = (14, 14)
    sz_a = (8, 8)
    sz_s = (376, 541)

    def prepare(self):
        for mode, lst in self.list.items():
            names = []
            h5 = h5py.File(self.get_h5_path(mode), 'w')
            for source, items in lst.items():
                for it in items:
                    h5_pth = "/%s/%s" % (source, it)
                    logger.info("Preparing sample '%s'." % h5_pth)
                    names.append(h5_pth)
                    if source == 'Stanford':
                        name = it.split('/')
                        pth = path.join(self.source_dir[source], name[0], 'raw', name[1]+'.png')
                    elif source == 'SIGASIA16':
                        pth = path.join(self.source_dir[source], it+'.png')
                    else:
                        raise NotImplementedError

                    rgb = load_item(pth_img=pth, a_raw=self.sz_a_raw, a_preserve=self.sz_a, ycbcr=False)
                    h5.create_dataset(h5_pth + '/rgb/original', data=np.transpose(rgb, (4, 0, 1, 2, 3)))

                    for scale in [2, 3, 4]:
                        sz_s = np.array(rgb.shape[-3:-1])
                        sz_s = sz_s // scale
                        lr = spatial_resize(rgb[:, :, :sz_s[0]*scale, :sz_s[1]*scale], sz_s)
                        h5.create_dataset(h5_pth + '/rgb/%dx' % scale, data=np.transpose(lr, (4, 0, 1, 2, 3)))

            h5.attrs['names'] = ','.join(names)
            h5.close()
