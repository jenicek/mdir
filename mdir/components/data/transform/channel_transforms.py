import numpy as np

from . import functional as tfunc
from .core_transforms import GenericTransform

#
# Generic
#

class AddConstantChannel(GenericTransform):
    """Add a constant channel to all images"""

    def __init__(self, value):
        super().__init__({"value": float(value)})

    def __call__(self, *pics):
        acc = []
        for pic in pics:
            assert isinstance(pic, np.ndarray)
            acc.append(np.concatenate((pic, np.full(pic.shape[:-1] + (1,), self.params["value"], dtype=np.float32)), axis=2))
        return acc


class NpInvertChannel(GenericTransform):
    """Invert channel of a numpy array, so that 0 becomes 1 and vice versa"""

    def __init__(self, channel):
        super().__init__({"channel": int(channel)})

    def __call__(self, *pics):
        for pic in pics:
            pic[:,:,self.params["channel"]] = 1 - pic[:,:,self.params["channel"]]
        return pics


class NpChanSelector(GenericTransform):
    """Keep only defined channels from a numpy array"""

    def __init__(self, start, end="unset"):
        if end != "unset":
            end = int(end) if end and end != "null" else None
        super().__init__({"start": int(start), "end": end})

    def __call__(self, *pics):
        if self.params['start'] == "unset":
            return [x[:,:,self.params['start']:self.params['start']+1] for x in pics]
        return [x[:,:,self.params['start']:self.params['end']] for x in pics]


class NpCloneChannels(GenericTransform):
    """Duplicate specified channels from a numpy array"""

    def __init__(self, start, end="unset"):
        if end != "unset":
            end = int(end) if end and end != "null" else None
        super().__init__({"start": int(start), "end": end})

    def __call__(self, *pics):
        if self.params['end'] == "unset":
            return [np.concatenate((x, x[:,:,self.params['start']:self.params['start']+1]), axis=2) for x in pics]
        return [np.concatenate((x, x[:,:,self.params['start']:self.params['end']]), axis=2) for x in pics]


#
# Intensity-related
#

class AddIntensityFromRgb(GenericTransform):
    """Add a lightness channel from its rgb channels"""

    def __init__(self, colorspace="lab"):
        super().__init__({"colorspace": colorspace})

    def __call__(self, *pics):
        acc = []
        for pic in pics:
            assert isinstance(pic, np.ndarray)
            spc = tfunc.rgb2normspace(pic[:,:,:3], self.params["colorspace"])
            acc.append(np.concatenate((pic, spc[:,:,:1]), axis=2))
        return acc


class ToColorspace(GenericTransform):
    """Convert ``numpy.ndarray`` from RGB to given colorspace."""

    def __init__(self, colorspace):
        super().__init__({"colorspace": colorspace})

    def __call__(self, *pics):
        return [tfunc.rgb2normspace(pic[:,:,:3], self.params["colorspace"]) for pic in pics]


#
# Edges-related
#

class AddEdgesDollarFromRgb(GenericTransform):
    """Add a channel that is the output of the Dollar's edge detector"""

    CLAHE_CLIPLIMIT = 4
    CLAHE_TILEGRIDSIZE = 8
    CLAHE_COLORSPACE = "lab"

    def __init__(self, model, resize=None, prefilter=None, postfilter=None):
        """
        Model: lsmodelBsds
        """
        resize = resize or None
        prefilter = prefilter or None
        postfilter = postfilter or None
        super().__init__({"model": model, "resize": resize, "prefilter": prefilter, "postfilter": postfilter})
        assert not resize
        assert prefilter in {None, "clahe"}
        assert postfilter in {None, "edgefilter"}

        self.prefilter = None
        if self.params["prefilter"] == "clahe":
            self.prefilter = tfunc.ImageClahe(clip_limit=self.CLAHE_CLIPLIMIT, grid_size=self.CLAHE_TILEGRIDSIZE, colorspace=self.CLAHE_COLORSPACE)
        self.detector = tfunc.EdgesDollar(self.params["model"])

    def __call__(self, *pics):
        acc = []
        for pic in pics:
            assert isinstance(pic, np.ndarray)
            input_pic = pic[:,:,:3]
            if self.prefilter:
                input_pic = self.prefilter.apply(input_pic)
            edges = self.detector.apply(input_pic)
            if self.params["postfilter"] == "edgefilter":
                edges = self.detector.cirsketch_edgefilter(edges)
            acc.append(np.concatenate((pic, np.expand_dims(edges, axis=2)), axis=2))
        return acc
