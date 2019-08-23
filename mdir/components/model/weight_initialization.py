import math
import torch
import torch.nn as nn


def init_weights_normal(m):
    """Initialize weights from normal distribution with mean=0 and std=1"""
    classname = m.__class__.__name__
    if classname == 'Conv2d':
        nn.init.normal_(m.weight.data)
        nn.init.normal_(m.bias.data)

def init_weights_normal_p2p(m):
    """Official pix2pix initialization implementation"""
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def _calculate_fan_in(tensor):
    """Calculate the number of input units of given tensor"""
    dimension = tensor.ndimension()
    if dimension < 2:
        raise ValueError("Fan in can not be computed for tensor with less than 2 dimensions")

    fan_in = tensor.size(1)
    if dimension > 2 and tensor.dim() > 2: # not Linear
        fan_in *= tensor[0][0].numel() # receptive field size
    return fan_in

def he_normal_(tensor):
    """HE normal initializer"""
    fan_in = _calculate_fan_in(tensor)
    std = math.sqrt(2.0 / fan_in)
    with torch.no_grad():
        return tensor.normal_(0, std)

def init_weights_he_normal(m):
    """Initialize weights using HE normal and biases using const, used in orig unet"""
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        he_normal_(m.weight.data)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.01) # Better than 0 if relu follows


WEIGHT_INITIALIZATIONS = {
    "normal": init_weights_normal,
    "normal_p2p": init_weights_normal_p2p,
    "he_normal": init_weights_he_normal,
}
