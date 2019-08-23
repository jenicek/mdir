import numpy as np
import torch
import torch.nn.functional as F

from ...tools.utils import load_path


class Compose(object):
    """Take multiple wrappers and apply them sequentially. In forward direction for preprocess,
        in backward direction for postprocess."""

    def __init__(self, wrappers, device):
        """Store a list of wrappers"""
        self.wrappers = wrappers
        self.device = device

    def __call__(self, tensor, inference, model=None):
        """Apply wrappers sequentially for preprocess, evaluate model and apply wrappers
            sequentially for postprocess in reversed order. Return tensor."""
        if not self.wrappers:
            return inference(tensor.to(self.device))
        if model is None:
            model = inference

        metadata = []
        for wrapper in self.wrappers:
            tensor, meta = wrapper.preprocess(tensor, model)
            metadata.append(meta)

        if isinstance(tensor, list):
            tensor = [inference(x.to(self.device)) for x in tensor]
        else:
            tensor = inference(tensor.to(self.device))

        for wrapper, meta in reversed(list(zip(self.wrappers, metadata))):
            tensor = wrapper.postprocess(tensor, model, meta)
        return tensor

    def __repr__(self):
        nice_wrappers = "\n" + "".join("    %s\n" % x for x in self.wrappers) if self.wrappers else ""
        return f"""{self.__class__.__name__}([{nice_wrappers}])"""


class Wrapper(object):
    """Wrap network input and output with custom functions to support various inference patterns
        such as tiling. This should be used only during inference"""

    def __init__(self, device):
        pass

    def preprocess(self, tensor, _model):
        """Return unmodified input tensor and None for metadata. Applied on network input."""
        return tensor, None

    def postprocess(self, tensor, _model, _metadata):
        """Return unmodified input tensor. Applied on network output."""
        return tensor


class ReflectPadMakeDivisible(Wrapper):
    """Pad tensor so that its spatial dimension is divisible by a specified number. Pad it using
        reflection around the boundary."""

    def __init__(self, divisible_by, device):
        """Store divisible_by int"""
        super().__init__(device)
        self.divisible_by = int(divisible_by) if isinstance(divisible_by, str) else divisible_by

    def preprocess(self, tensor, _model):
        """Return padded tensor and used padding"""
        size = np.array(tensor.size())[2:]
        padx, pady = (np.ceil(size / self.divisible_by)*self.divisible_by - size) / 2
        padding = (int(np.floor(pady)), int(np.ceil(pady)), int(np.floor(padx)), int(np.ceil(padx)))
        return F.pad(tensor, padding, 'replicate'), padding

    def postprocess(self, tensor, _model, padding):
        """Return cropped tensor based on given padding"""
        return tensor[:,:,padding[2]:-padding[3] or None,padding[0]:-padding[1] or None]

    def __repr__(self):
        return f"{self.__class__.__name__} (divisible_by={self.divisible_by})"


class CirMultiscaleAggregation(Wrapper):
    """Downscale each image to defined scales and aggregate resulting descriptors."""

    def __init__(self, scales, device):
        """Parse and store scales"""
        super().__init__(device)
        if isinstance(scales, str):
            scales = {"True": True, "False": False}[scales]
        if isinstance(scales, bool):
            scales = [1, 1./np.sqrt(2), 1./2] if scales else [1]
        self.scales = scales

    def preprocess(self, tensor, _model):
        if len(self.scales) == 1:
            return tensor if isinstance(tensor, list) else [tensor], isinstance(tensor, list)

        acc = []
        if isinstance(tensor, list):
            for single in tensor:
                for scale in self.scales:
                    acc.append(F.interpolate(single, scale_factor=scale, mode='bilinear', align_corners=False))
            return acc, True

        return [F.interpolate(tensor, scale_factor=scale, mode='bilinear', align_corners=False) for scale in self.scales], False

    @staticmethod
    def aggregate_tensor(tensor, nscales, outputdim, msp):
        assert len(tensor) == nscales, "%s != %s" % (len(tensor), nscales)
        v = torch.zeros(outputdim, dtype=tensor[0].dtype, device=tensor[0].device)
        for subtensor in tensor:
            v += subtensor.pow(msp).squeeze()

        v = (v / nscales).pow(1./msp)
        v /= v.norm()

        return v

    def postprocess(self, tensor, model, waslist):
        msp = 1
        if len(self.scales) > 1 and model.meta['pooling'] == 'gem' and not model.meta['regional'] and not model.meta['whitening']:
            msp = model.pool.p.item()

        if not waslist:
            return self.aggregate_tensor(tensor, len(self.scales), model.meta['out_channels'], msp)

        assert len(tensor) % len(self.scales) == 0, "%s %% %s != 0" % (len(tensor), len(self.scales))
        acc = []
        for i in range(0, len(tensor), len(self.scales)):
            acc.append(self.aggregate_tensor(tensor[i:i+len(self.scales)], len(self.scales), model.meta['out_channels'], msp))
        return acc

    def __repr__(self):
        return f"{self.__class__.__name__}(scales={self.scales})"


class FakeBatch(Wrapper):
    """Mimic batch behaviour by accumulating the result across multiple images"""

    def postprocess(self, tensor, model, _meta):
        if not isinstance(tensor, list):
            return tensor

        output = torch.zeros(model.meta['out_channels'], len(tensor), device=tensor[0].device)
        for j, vec in enumerate(tensor):
            output[:, j] = vec.squeeze()
        return output

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class CirFakeTupleBatch(FakeBatch):
    """Mimic batch behaviour by accumulating the result across multiple images in multiple tuples"""

    @classmethod
    def unsqueeze(cls, tensor):
        if isinstance(tensor, list):
            return [cls.unsqueeze(x) for x in tensor]
        elif len(tensor.shape) == 3:
            return tensor.unsqueeze_(0)
        elif len(tensor.shape) == 4:
            return tensor
        raise ValueError("Unsupported tensor dimensionality %s" % len(tensor.shape))

    def preprocess(self, tensor, _model):
        """Flatten the 2d list"""
        if not isinstance(tensor, list) or not isinstance(tensor[0], list):
            return tensor, False

        acc = []
        meta = len(tensor[0])
        for tpl in tensor:
            assert meta == len(tpl)
            acc += tpl
        return acc, meta


class CirtorchWhiten(Wrapper):
    """Whiten vectors with possible dimensionality reduction"""

    def __init__(self, whitening, dimensions, device):
        """Load whitening by its whitening path and store dimensions for reduction (empty for
            disable)"""
        super().__init__(device)
        whitening = load_path(whitening)
        self.P = torch.tensor(whitening['P'], dtype=torch.float32, device=device)
        self.m = torch.tensor(whitening['m'], dtype=torch.float32, device=device)
        self.dimensions = dimensions or self.P.shape[0]

    def postprocess(self, tensor, model, _meta):
        X = self.P[:self.dimensions, :].mm(tensor.unsqueeze_(1).sub_(self.m))
        return X.div_(torch.norm(X, p=2, dim=0, keepdim=True) + 1e-6).squeeze()


WRAPPERS_LABELS = {
    "reflectpad_divisible": ReflectPadMakeDivisible,
    "cirmultiscale": CirMultiscaleAggregation,
    "fakebatch": FakeBatch,
    "cirfaketuplebatch": CirFakeTupleBatch,
    "cirwhiten": CirtorchWhiten,
}


# Init function

def initialize_wrappers(net_wrappers, device):
    if net_wrappers is None:
        wraps = []
    elif isinstance(net_wrappers, str):
        wraps = []
        for wrap in [x for x in net_wrappers.split(",") if x]:
            wname, *args = wrap.split(":", 1)
            args = args[0].split(",") if args else []
            wraps.append(WRAPPERS_LABELS[wname](*args, device=device))
    else:
        wraps = [WRAPPERS_LABELS[x.split("_", 1)[1]](**net_wrappers[x], device=device) for x in sorted(net_wrappers)]
    return Compose(wraps, device)
