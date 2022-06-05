import logging
import sys

__all__ = ['logger', 'set_log_file']


fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=fmt)
logger = logging.getLogger()


def set_log_file(pth):
    handler = logging.FileHandler(pth, "a")
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
