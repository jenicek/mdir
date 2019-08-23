import os.path
import copy
import numpy as np
import torch
from torch import nn
from collections import OrderedDict
from cirtorch.networks.imageretrievalnet import init_network, ImageRetrievalNet
from cirtorch.utils.general import get_root

def init_cirnet(**params):
    for key in ["local_whitening", "pooling", "regional", "whitening", "pretrained"]:
        if key not in params:
            raise ValueError("Key '%s' not in params" % key)
    params['mean'] = [0.485, 0.456, 0.406]
    params['std'] = [0.229, 0.224, 0.225]
    params['model_dir'] = os.path.join(get_root(), "weights")
    params['architecture'] = params.pop("cir_architecture")

    net = init_network(params)
    net.meta["in_channels"] = 3
    net.meta["out_channels"] = net.meta["outputdim"]
    return net


class ImageRetrievalNetBranched(ImageRetrievalNet):

    def __init__(self, branches, merging, aggregation, features, lwhiten, pool, whiten, meta):
        super().__init__(features, lwhiten, pool, whiten, meta)
        assert len(branches) > 1
        self.branches = nn.ModuleDict([(x, nn.Sequential(*y)) for x, y in branches.items()])
        self.ranges = np.cumsum([0] + [x for x, y in merging])
        self.weights = [y for x, y in merging]
        self.aggregation = aggregation

    def forward(self, x):
        # x -> features
        acc = [self.weights[i] * y(x[:,self.ranges[i]:self.ranges[i+1]]) for i, y in enumerate(self.branches.values())]
        if self.aggregation == "sum":
            acc = torch.sum(torch.stack(acc), dim=0)
        else:
            acc = torch.cat(acc, dim=1)

        return super().forward(acc)


def init_cirnet_branched(**params):
    """
    channels:
        branches:
            0_rgb: {in: 3, init: clone, weight: 1}
            1_gray: {in: 1, init: avg, weight: 1}
        merge:
            layer: 3
            aggregation: sum | concat
    """
    channels = params.pop("channels")
    model = init_cirnet(**params)

    # Merge
    merge = channels.pop("merge")
    assert merge.keys() == {"layer", "aggregation"}, merge.keys()

    if merge["layer"] > 0:
        split_idx = [i for i, x in enumerate(model.features) if isinstance(x, nn.Conv2d)][merge["layer"]]
        features_pre = model.features[:split_idx]
        features_post = model.features[split_idx:]
        assert features_pre[0].in_channels == 3

        # Branches
        branches = OrderedDict()
        merging = []
        for key, branch in sorted(channels["branches"].items()):
            assert branch.keys() == {"in", "init", "weight"}, branch.keys()
            assert branch["init"] in {"sum", "clone"}, branch["init"]

            features = copy.deepcopy(features_pre)
            features[0].in_channels = branch["in"]

            if branch["init"] == "sum":
                assert branch["in"] == 1
                features[0].weight.data = features[0].weight.data.sum(dim=1, keepdim=True)
            elif branch["init"] == "clone":
                assert branch["in"] == 3
            branches[key.split("_", 1)[1]] = features
            merging.append((branch["in"], branch["weight"]))

        # Merge
        if merge["aggregation"] == "concat":
            features_post[0].in_channels *= len(branches)
            features_post[0].weight.data = features_post[0].weight.data.repeat(1, len(branches), 1, 1)
        elif merge["aggregation"] != "sum":
            raise ValueError("Unsupported aggregation %s" % merge["aggregation"])

        model.meta["in_channels"] = sum([x for x, y in merging])
        return ImageRetrievalNetBranched(branches, merging, merge["aggregation"], features_post, model.lwhiten, model.pool, model.whiten, model.meta)

    else:
        assert merge["layer"] == 0
        assert merge["aggregation"] == "concat"

        acc = []
        for key, branch in sorted(channels["branches"].items()):
            assert branch.keys() == {"in", "init", "weight"}, branch.keys()
            assert branch["init"] in {"sum", "clone"}, branch["init"]

            if branch["init"] == "sum":
                assert branch["in"] == 1
                acc.append(branch["weight"] * model.features[0].weight.data.clone().sum(dim=1, keepdim=True))
            elif branch["init"] == "clone":
                assert branch["in"] == 3
                acc.append(branch["weight"] * model.features[0].weight.data.clone())

        model.features[0].in_channels = sum(x["in"] for x in channels["branches"].values())
        model.features[0].weight.data = torch.cat(acc, dim=1)
        model.meta["in_channels"] = model.features[0].in_channels
        return model
