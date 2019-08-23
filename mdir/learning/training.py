import copy
import time
import numpy as np
import torch

from .epoch_iteration import initialize_epoch_iteration
from ..tools.utils import indent
from ..components import optim


class EpochTraining:

    def __init__(self, params, criterion, optimizer, scheduler, epoch_iteration, epoch, *, epochs,
                 deterministic, seed):
        self.params = params
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch_iteration = epoch_iteration
        self.epoch = epoch

        self._epochs = epochs
        self.deterministic = deterministic
        self.seed = seed

        self.set_seed(seed if seed is not None else time.time())
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def __next__(self):
        self.epoch += 1
        if self.epoch >= self._epochs:
            raise StopIteration()

        if self.seed is not None:
            self.set_seed(self.epoch + self.seed)
        self.scheduler.step()
        return self.epoch, self.epoch_iteration.steps(self.epoch)

    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

    @property
    def remains_epochs(self):
        return self._epochs - self.epoch - 1

    @classmethod
    def initialize(cls, params, network, data, params_data, device, epoch, optimizer_state=None):
        store_params = copy.deepcopy(params)

        # Criterion
        criterion = optim.criterion.initialize_criterion(params.pop("criterion"))
        if criterion and device:
            criterion.to(device)

        # Optimizer
        optimizer_opts = params.pop("optimizer")
        optimizer = optim.optimizer.initialize_optimizer(network=network, params=optimizer_opts)
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)

        # Scheduler
        scheduler = optim.scheduler.initialize_scheduler(optimizer=optimizer, params=params.pop('scheduler'),
                                                         nepochs=params["epochs"], last_epoch=epoch)

        # Epoch iteration
        net_defaults = network.network_params.runtime.get("data", {})
        epoch_iteration = initialize_epoch_iteration(params.pop("epoch_iteration"), data=data, params_data=params_data,
                                                     default_criterion=criterion, net_defaults=net_defaults)

        return cls(store_params, criterion, optimizer, scheduler, epoch_iteration, epoch, **params)


    #
    # Load and save
    #

    def state_dict(self):
        return {
            "type": self.__class__.__name__,
            "params": self.params,
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": self.epoch,
        }

    @classmethod
    def initialize_from_state(cls, state_dict, network, data, params_data, device, params):
        assert state_dict["type"] == cls.__name__, state_dict["type"]
        assert state_dict["epoch"] < state_dict["params"]["epochs"]
        if params is not None:
            state_params_noe = {x: y for x, y in state_dict["params"].items() if x != "epochs"}
            params_noe = {x: y for x, y in params.items() if x != "epochs"}
            assert state_params_noe == params_noe, "%s != %s" % (str(state_params_noe), str(params_noe))
            state_dict["params"]["epochs"] = params["epochs"]

        training = cls.initialize(state_dict["params"], network, data, params_data, device, state_dict["epoch"],
                                  optimizer_state=state_dict['optimizer_state'])
        return training

    def __repr__(self):
        nice_params = "\n" + "".join("    %s: %s,\n" % (x, y) for x, y in self.params.items())

        return \
f"""{self.__class__.__name__} (
    params: {{{indent(str(nice_params))}}}
    optimizer: {indent(str(self.optimizer), 2)}
    scheduler: {indent(str(self.scheduler))}
    epoch_iteration: {indent(str(self.epoch_iteration))}
    epoch: {self.epoch}
)"""


# Initialization

TRAININGS = {
    "EpochTraining": EpochTraining,
}

def initialize_training(params, network, data, params_data, device, state=None):
    training_cls = params.pop("type")
    if state is None:
        return TRAININGS[training_cls].initialize(params, network, data, params_data, device, -1)
    else:
        return TRAININGS[training_cls].initialize_from_state(state, network, data, params_data, device, params)
