from . import base_optimizers, optimizer_compositions

def initialize_optimizer(network, params):
    if not params:
        return None

    if "composition" in params:
        return optimizer_compositions.initialize_optimizer_composition(network=network, params=params)

    return base_optimizers.initialize_base_optimizer(net_parameters=network.parameters(params), params=params)
