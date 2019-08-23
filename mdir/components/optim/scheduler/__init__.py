from . import base_schedulers, scheduler_compositions


def initialize_scheduler(optimizer, params, nepochs, last_epoch=-1):
    if not optimizer or not params:
        return None

    if "composition" in params:
        return scheduler_compositions.initialize_scheduler_composition(optimizer=optimizer, last_epoch=last_epoch, nepochs=nepochs, params=params)

    return base_schedulers.initialize_base_scheduler(optimizer=optimizer, last_epoch=last_epoch, nepochs=nepochs, params=params)
