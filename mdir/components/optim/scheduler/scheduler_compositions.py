from .base_schedulers import initialize_base_scheduler
from ....tools.utils import indent


class SchedulerSet:
    def __init__(self, schedulers):
        self.schedulers = schedulers

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    @classmethod
    def initialize(cls, optimizer, last_epoch, nepochs, scheduler_params):
        acc = []
        for net in optimizer:
            acc.append(initialize_base_scheduler(optimizer=optimizer[net], last_epoch=last_epoch, nepochs=nepochs, params=scheduler_params[net]))
        return cls(acc)

    def __repr__(self):
        return \
f"""{self.__class__.__name__} (
    schedulers: {indent(str(self.schedulers))}
)"""


# Initialization

SCHEDULER_COMPOSITIONS = {
    "set": SchedulerSet,
}

def initialize_scheduler_composition(optimizer, last_epoch, nepochs, params):
    return SCHEDULER_COMPOSITIONS[params["composition"].pop("type")].initialize(optimizer=optimizer, last_epoch=last_epoch, nepochs=nepochs, scheduler_params=params, **params.pop("composition"))
