from torch import optim


def init_sgd(net_parameters, lr, momentum, weight_decay):
    return optim.SGD(net_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)

def init_adam(net_parameters, lr, weight_decay):
    return optim.Adam(net_parameters, lr=lr, weight_decay=weight_decay)


BASE_OPTIMIZERS = {
    "sgd": init_sgd,
    "adam": init_adam,
}

def initialize_base_optimizer(net_parameters, params):
    return BASE_OPTIMIZERS[params.pop("algorithm")](net_parameters, **params)
