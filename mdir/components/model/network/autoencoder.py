import torch.nn as nn


class Identity(nn.Module):
    """Returns its input unaltered"""

    def __init__(self):
        super().__init__()
        self.meta = {}

    def forward(self, x):
        return x

class PixelConvRegr(nn.Module):
    def __init__(self, in_channels, out_channels, hidden, hidden_activation="relu"):
        super().__init__()
        self.meta = {"in_channels": in_channels, "out_channels": out_channels}
        hidden_activation = {"relu": nn.ReLU, "tanh": nn.Tanh}[hidden_activation]
        layers = []
        for inch, outch in zip([in_channels] + hidden, hidden):
            layers.append(nn.Conv2d(inch, outch, 1))
            layers.append(hidden_activation())
        layers.append(nn.Conv2d(hidden[-1], out_channels, 1))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class AutoencoderRegr(nn.Module):
    def __init__(self, in_channels, out_channels, hidden, hidden_activation="relu", reception_field=3):
        super().__init__()
        assert reception_field % 2 == 1
        self.meta = {"in_channels": in_channels, "out_channels": out_channels}
        hidden_activation = {"relu": nn.ReLU, "tanh": nn.Tanh}[hidden_activation]
        layers = []
        layers.append(nn.Conv2d(in_channels, hidden[0], reception_field, padding=reception_field//2))
        layers.append(hidden_activation())
        for inch, outch in zip(hidden, hidden[1:]):
            layers.append(nn.Conv2d(inch, outch, 1))
            layers.append(hidden_activation())
        layers.append(nn.Conv2d(hidden[-1], out_channels, 1))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PixelConvRes(nn.Module):
    def __init__(self, in_channels, out_channels, hidden, hidden_activation="relu"):
        super().__init__()
        self.meta = {"in_channels": in_channels, "out_channels": out_channels}
        hidden_activation = {"relu": nn.ReLU, "tanh": nn.Tanh}[hidden_activation]
        layers = []
        for inch, outch in zip([in_channels] + hidden, hidden):
            layers.append(nn.Conv2d(inch, outch, 1))
            layers.append(hidden_activation())
        layers.append(nn.Conv2d(hidden[-1], out_channels, 1))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x) + x[:,0:2]
