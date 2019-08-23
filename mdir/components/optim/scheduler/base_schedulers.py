import math
from torch.optim import lr_scheduler


class VoidScheduler:

    def step(self):
        pass


def init_void_scheduler(_optimizer, _last_epoch, _nepochs):
    return VoidScheduler()

def init_lambda_scheduler(optimizer, last_epoch, nepochs, fixed_ratio):
    """First, have fixed lr, then, decay it linearly to zero"""
    # Fixed ratio is e.g. 0.5
    def lambda_rule(epoch):
        return 1 - max(0, epoch + 1 - fixed_ratio*nepochs) / float((1-fixed_ratio)*nepochs + 1)
    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)

def init_gamma_scheduler(optimizer, last_epoch, _nepochs, gamma):
    # Gamma is e.g. 0.99 ~ exp(-0.01)
    if isinstance(gamma, str) and gamma.startswith("exp(") and gamma[-1] == ")":
        gamma = math.exp(float(gamma[len("exp("):-1]))

    return lr_scheduler.ExponentialLR(optimizer, gamma=gamma, last_epoch=last_epoch)


# Initialization

BASE_SCHEDULERS = {
    "const": init_void_scheduler,
    "lambda": init_lambda_scheduler,
    "gamma": init_gamma_scheduler,
}

def initialize_base_scheduler(optimizer, last_epoch, nepochs, params):
    return BASE_SCHEDULERS[params.pop('algorithm')](optimizer, last_epoch, nepochs, **params)
