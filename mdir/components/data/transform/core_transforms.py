import numpy as np
import torch
from torchvision import transforms


#
# Abstract & Lib
#

class GenericTransform(object):

    def __init__(self, params=None):
        self.params = params or {}

    def __repr__(self):
        return self.__class__.__name__ + '(%s)' % ", ".join("%s=%s" % (x, str(y)) for x, y in self.params.items())


#
# Core
#

class Compose(transforms.Compose):

    def __call__(self, *pics): # pylint: disable=arguments-differ
        for t in self.transforms:
            pics = t(*pics)
        if len(pics) == 1:
            return pics[0]
        return pics


class ToTensor(transforms.ToTensor):

    def __call__(self, *pics): # pylint: disable=arguments-differ
        return [super(ToTensor, self).__call__(x) for x in pics]


class Normalize(GenericTransform):

    def __init__(self, mean, std, strict_shape=True):
        strict_shape = bool(strict_shape) if not isinstance(strict_shape, str) or strict_shape.lower() != "false" else False
        super().__init__({"mean": mean, "std": std, "strict_shape": strict_shape})
        assert len(mean) == len(std)

    def __call__(self, *pics):
        acc = []
        for pic in pics:
            if self.params["strict_shape"]:
                assert pic.size(0) == len(self.params["mean"]), (pic.size(0), len(self.params["mean"]))
                acc.append(transforms.functional.normalize(pic, self.params["mean"], self.params["std"]))
            else:
                assert pic.size(0) <= len(self.params["mean"]), (pic.size(0), len(self.params["mean"]))
                acc.append(transforms.functional.normalize(pic, self.params["mean"][:pic.size(0)], self.params["std"][:pic.size(0)]))

        return acc


class Pil2Numpy(GenericTransform):
    """Convert pil image to numpy array with values between 0 and 1"""

    def __call__(self, *pics):
        return [np.array(x.convert('RGB'), dtype=np.float32)/255.0 for x in pics]


class StackBatch(GenericTransform):
    """Convert a list of image tensors to a single tensor by concatenating them along the axis 0"""

    def __call__(self, *pics):
        return [torch.cat(pics, 0)]


class NanCheck(GenericTransform):
    """Check for nan in images, raise an exception when nan detected. Return input unchanged"""

    def __call__(self, *pics):
        for pic in pics:
            if np.isnan(pic).any():
                raise ValueError("Nan value occured in input")
        return pics
