import inspect
import json

from .CustomConfig import CustomConfig

__all__ = ['CustomConfig']


def get_properties(cls):
    props = [attr for attr in inspect.getmembers(cls) if attr[0][:1] != '_']
    data = {}
    for prop in props:
        k, v = prop[0], prop[1]
        if isinstance(v, type):
            v = get_properties(v)
        if not inspect.ismethod(v):
            data[k] = v
    return data


def config_to_json(config, pth):
    with open(pth, 'w', encoding='utf-8') as f:
        json.dump(get_properties(config), f, ensure_ascii=False, indent=4, skipkeys=True)


