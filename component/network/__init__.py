from .DKNet import DKNet, DKNetConfig

__all__ = ['create_model']


def create_model(config):
    dic = {
        'DKNet': (DKNet, DKNetConfig),
    }
    if config.net not in dic.keys():
        raise ValueError("Unknown network.")
    net, n_config = dic[config.net]
    config.net_config = n_config
    model = net(config)

    return model
