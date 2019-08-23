from torch import nn

class L1Loss(nn.L1Loss):

    def __init__(self):
        super().__init__(reduction="mean")


class MSELoss(nn.MSELoss):

    def __init__(self):
        super().__init__(reduction="mean")
