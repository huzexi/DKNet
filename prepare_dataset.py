from component.config import CustomConfig as config
from component.dataset import get_dataset, BaseDataset
from component.log import logger

if __name__ == '__main__':
    lst = {
        'Yeung',
    }
    for ds in lst:
        logger.info("Start to prepare %s dataset." % ds)
        dataset = get_dataset(ds, config, BaseDataset.MODE_PREPARE)
        dataset.prepare()
