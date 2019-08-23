import numpy as np
from . import functional as tfunc

from .core_transforms import GenericTransform

#
# Clahe
#

class AddClaheFromRgb(GenericTransform):
    """Add a channel that is image's clahe-normalized lightness taken from its rgb channels"""

    def __init__(self, clip_limit=4, grid_size=8, colorspace="lab"):
        super().__init__({"clip_limit": int(clip_limit), "grid_size": grid_size, "colorspace": colorspace})
        self.clahe = tfunc.ChannelClahe(clip_limit=int(clip_limit), grid_size=grid_size)

    def __call__(self, *pics):
        acc = []
        for pic in pics:
            assert isinstance(pic, np.ndarray)
            spc = tfunc.rgb2normspace(pic[:,:,:3], self.params["colorspace"])
            chan = self.clahe.apply(spc[:,:,0])
            pic0 = np.concatenate((pic, np.expand_dims(chan, axis=2)), axis=2)
            acc.append(pic0)
        return acc


class ApplyClahe(GenericTransform):
    """Convert input image to given colorspace and apply clahe to its lightness channel."""

    def __init__(self, clip_limit=4, colorspace="lab", grid_size=8):
        super().__init__({"clip_limit": clip_limit, "colorspace": colorspace, "grid_size": grid_size})
        self.clahe = tfunc.ImageClahe(**self.params)

    def __call__(self, pic):
        return [self.clahe.apply(pic)]


class CreateClahedImage(ApplyClahe):
    """Add a second CLAHE-normalized image"""

    def __call__(self, pic):
        return [pic, self.clahe.apply(pic[:,:,:3])]


#
# Match histogram
#

class MatchHistogram(GenericTransform):
    """Convert input image to given colorspace and match its lightness channel histogram to given
        value."""

    def __init__(self, histogram, colorspace="lab"):
        super().__init__({"histogram": histogram, "colorspace": colorspace})

    def __call__(self, pic):
        return [tfunc.image_histogram_matching(pic, **self.params)]


class ReplaceChannelWithHistogram(GenericTransform):
    """Add a channel to the first image which is its last channel transformed to have specific
        pixel histogram. In training time, the histogram is computed from the last channel
        of the second image and this is removed. In test time, the histogram is a constant."""

    def __init__(self, histogram, created_channel):
        super().__init__({"histogram": histogram, "created_channel": created_channel})
        assert created_channel in {"append", "replace"}

    def __call__(self, pic0, pic1=None):
        pic0_output = pic0[:,:,:-1] if self.params["created_channel"] == "replace" else pic0

        if pic1 is not None:
            # Histogram matching with ground-truth image
            add_chan = tfunc.channel2channel_histogram_matching(pic0[:,:,-1], pic1[:,:,-1])
            return np.concatenate((pic0_output, np.expand_dims(add_chan, axis=2)), axis=2), pic1[:,:,:-1]
        else:
            # Histogram matching with pre-defined histogram
            add_chan = tfunc.channel_histogram_matching(pic0[:,:,-1], self.params["histogram"])
            return np.concatenate((pic0_output, np.expand_dims(add_chan, axis=2)), axis=2),


#
# Gamma equalize
#

class GammaEqualize(GenericTransform):
    """Convert image to defined colorspace and apply gamma to the lightness channel, so that mean
        is shifted to target value between zero and one."""

    def __init__(self, target, colorspace="lab"):
        target = float(target)
        super().__init__({"target": target, "colorspace": colorspace})
        assert target > 0 and target < 1, target

    def __call__(self, pic):
        return [tfunc.image_gamma_matching(pic, **self.params)]
