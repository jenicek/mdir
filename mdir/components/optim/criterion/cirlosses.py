import torch
from cirtorch.layers import loss as cirloss

class ContrastiveLoss(cirloss.ContrastiveLoss):
    def __init__(self, margin, eps):
        super().__init__(margin=margin, eps=eps)
        self.reduction = "sum"

    def forward(self, x, label):
        if isinstance(label, list):
            label = torch.cat(label)
        return super().forward(x, label.to(x.device))

class TripletLoss(cirloss.TripletLoss):
    def __init__(self, margin):
        super().__init__(margin=margin)
        self.reduction = "sum"

    def forward(self, x, label):
        if isinstance(label, list):
            label = torch.cat(label)
        return super().forward(x, label.to(x.device))
