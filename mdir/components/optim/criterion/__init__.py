from torch import nn
from . import base_losses, cirlosses

CRITERIA = {
    "l1": base_losses.L1Loss,
    "mse": base_losses.MSELoss,
    "contrastive": cirlosses.ContrastiveLoss,
    "triplet": cirlosses.TripletLoss,
}

def initialize_criterion(params):
    if not params:
        return None

    return CRITERIA[params.pop("loss")](**params)
