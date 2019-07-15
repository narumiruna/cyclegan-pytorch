from pprint import pprint

from cyclegan.utils import AttrDict, load_yaml


def load_config(f=None):
    config = AttrDict()

    # training config
    config.train = AttrDict()
    config.train.num_epochs = 20
    config.train.batch_size = 32

    # optimizer config
    config.optimizer = AttrDict()
    config.optimizer.name = 'SGD'
    config.optimizer.learning_rate = 1e-3

    # load from yaml
    if f is not None:
        data = load_yaml(f)
        config.update(data)

    # set config to immutable
    config.set_immutable()

    # print config
    pprint(config)

    return config
