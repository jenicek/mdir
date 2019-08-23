import torch.nn as nn

from . import autoencoder, unet, cirnet


class Identity(nn.Module):
    """Returns its input unaltered"""

    def __init__(self):
        super().__init__()
        self.meta = {}

    def forward(self, x):
        return x


MODEL_LABELS = {
    "identity": Identity,
    "orig_unet": unet.OrigUNet,
    "p2p_unet": unet.P2pUNet,
    "outconv_unet": unet.OutconvP2pUNet,
    "outconv_dynint_unet": unet.OutconvP2pUNetDynamicInterpolate,

    "shallow_p2p_unet": unet.ShallowP2pUNet,
    "inconv_p2p_unet": unet.InconvP2pUNet,
    "aligned_p2p_unet": unet.AlignedP2pUNet,

    "pixelconv_regr": autoencoder.PixelConvRegr,
    "pixelconv_res": autoencoder.PixelConvRes,
    "autoencoder_regr": autoencoder.AutoencoderRegr,

    "cirnet": cirnet.init_cirnet,
    "cirnet_branched": cirnet.init_cirnet_branched,
}

def initialize_model(params):
    return MODEL_LABELS[params.pop("architecture")](**params)
