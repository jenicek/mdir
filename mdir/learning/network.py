import os
import copy
import time
from collections import namedtuple
import abc
import torch

from ..components.model import network, weight_initialization
from ..components.data.wrapper import initialize_wrappers
from ..tools.utils import indent, load_url


class Network(abc.ABC):

    TRAIN = "train"
    EVAL = "eval"

    def __init__(self, frozen, model=None):
        self.stage = None
        self.frozen = frozen
        self.model = model
        if frozen:
            self.eval()

    @staticmethod
    def initialize_wrappers(wrappers, device):
        if isinstance(wrappers, dict):
            assert wrappers.keys() == {"train", "eval"}, wrappers.keys()
            return {x: initialize_wrappers(wrappers[x], device) for x in wrappers}

        return {x: initialize_wrappers(wrappers, device) for x in ["train", "eval"]}

    def train(self):
        if not self.frozen:
            self.model.train()
            self.stage = Network.TRAIN
        return self

    def eval(self):
        self.model.eval()
        self.stage = Network.EVAL
        return self

    def freeze(self, net="net"):
        assert net == "net"
        self.frozen = True
        self.eval()
        return self

    def parameters(self, _optimizer_opts, net="net"):
        assert net == "net"
        if self.frozen:
            return []
        return self.model.parameters()

    # Debug data functions

    def train_data(self):
        return [{"key": "net/params", "dtype": "weight/param", "data": dict(self.model.named_parameters())}]

    def const_data(self):
        acc = []
        graph = self.generate_network_graph()
        if graph is not None:
            acc.append({"key": "network_graph", "dtype": "blob", "data": {"net": {"dtype": "image:rgb", "data": graph}}})
        return acc

    def generate_network_graph(self):
        return None


class SingleNetwork(Network):

    NetworkParams = namedtuple("NetworkParams", ["model", "runtime"])

    def __init__(self, model, network_params, device, frozen):
        self.meta = {"in_channels": model.meta["in_channels"], "out_channels": model.meta["out_channels"]}
        self.network_params = network_params
        self.wrappers = self.initialize_wrappers(network_params.runtime.get("wrappers", ""), device)
        super().__init__(network_params.runtime.get("frozen", False) or frozen, model.to(device))

        # Limit to supported keys
        assert not network_params.runtime.keys() - {"data", "wrappers", "frozen"}, \
            network_params.runtime.keys() - {"data", "wrappers", "frozen"}
        assert not network_params.runtime.get("data", {}).keys() - {"mean_std", "transforms"}, \
            network_params.runtime.get("data", {}).keys() - {"mean_std", "transforms"}

    def __call__(self, image):
        return self.wrappers[self.stage](image, self.model)

    @classmethod
    def initialize(cls, params, device):
        # Initialize model
        path = params.pop("path", None)
        if not path:
            network_params = cls.NetworkParams(params.pop("model"),
                                               params.pop("runtime"))
            model = network.initialize_model(copy.deepcopy(network_params.model))
            init = params.pop("initialize")
            if init and init["weights"] != "default":
                torch.manual_seed(init["seed"] if init["seed"] is not None else time.time())
                model.apply(weight_initialization.WEIGHT_INITIALIZATIONS[init["weights"]])
        else:
            # Pretrained model
            print(">> Loaded net from %s" % path)
            if path.startswith("http://") or path.startswith("https://"):
                path = load_url(path)
            checkpoint = torch.load(path, map_location=lambda storage, location: storage)
            # Handle runtime inheritance
            runtime = params.pop("runtime")
            if runtime == "load_from_checkpoint":
                runtime = checkpoint["network_params"]["runtime"]
            else:
                runtime = {x: y if y != "load_from_checkpoint" else checkpoint["network_params"]["runtime"][x] for x, y in runtime.items()}
            # Initialize
            network_params = cls.NetworkParams(checkpoint["network_params"]["model"], runtime)
            model = network.initialize_model(copy.deepcopy(network_params.model))
            model.load_state_dict(checkpoint['model_state'])
            # Can be optionally provided
            params.pop("initialize", None)
            if "model" in params:
                assert params.pop("model") == checkpoint["network_params"]["model"]

        assert not params, params.keys()

        return cls(model, network_params, device=device, frozen=False)

    def overlay_params(self, new_params, device):
        if not new_params:
            return self

        new_params["runtime"]["frozen"] = True
        network_params = self.NetworkParams(self.network_params.model,
                                            new_params.pop("runtime"))
        assert not new_params
        return self.__class__(self.model, network_params, device, frozen=True)

    #
    # Load and save
    #

    def state_dict(self):
        return {
            "net": {
                "type": self.__class__.__name__,
                "frozen": self.frozen,
                "network_params": self.network_params._asdict(),
                "model_state": self.model.state_dict(),
            }
        }

    @classmethod
    def initialize_from_state(cls, state_dict, device, params, runtime):
        assert state_dict.keys() == {"net"}, state_dict.keys()
        checkpoint = state_dict["net"]
        assert checkpoint.keys() == {"type", "frozen", "network_params", "model_state"}, checkpoint.keys()
        network_params = cls.NetworkParams(**checkpoint["network_params"])

        assert checkpoint["type"] == cls.__name__, checkpoint["type"]
        if params is not None and "path" not in params:
            del params["initialize"]
            assert network_params._asdict() == params, "%s != %s" % (str(network_params._asdict()), str(params))

        model = network.initialize_model(copy.deepcopy(network_params.model))
        model.load_state_dict(checkpoint['model_state'])

        if runtime:
            network_params.runtime.update(runtime)

        return cls(model, network_params, device=device, frozen=checkpoint["frozen"])

    #
    # Utils
    #

    def generate_network_graph(self):
        # Bit of a hack
        import random
        import imageio
        from torchviz.dot import make_dot

        model = network.initialize_model(copy.deepcopy(self.network_params.model))
        x_in = torch.zeros(1, self.model.meta.get("in_channels", 3), 512, 512, requires_grad=True)
        fname = '/tmp/jenicto2.network_graph.%s' % random.randint(0, 1000000)
        y_pred = model(x_in)
        make_dot(y_pred, params=dict(list(model.named_parameters()) + [('x', x_in)])).render(fname, cleanup=True)
        img = imageio.imread(fname+".png")
        os.remove(fname+".png")
        return img

    def __repr__(self):
        nice_params = "\n" + "".join("    %s: %s,\n" % (x, y) for x, y in self.network_params._asdict().items())
        nice_wrappers = "\n" + "".join("    %s: %s,\n" % (x, indent(str(y))) for x, y in self.wrappers.items())

        return \
f"""{self.__class__.__name__} (
    meta: {self.meta}
    model: {indent(str(self.model))}
    network_params: {{{indent(nice_params)}}}
    wrappers: {{{indent(nice_wrappers)}}}
)"""


class SequentialNetwork(Network):

    NetworkParams = namedtuple("NetworkParams", ["runtime"])

    def __init__(self, networks, sequence, device, frozen):
        assert len(networks) == len(sequence)
        assert len(networks) == 2 # Currently tested only for a sequence of 2 networks
        self.sequence = sequence
        self.networks = networks
        first_net = networks[sequence[0]]
        last_net = networks[sequence[1]]
        super().__init__(frozen, last_net.model)

        # Handle wrappers
        self.wrappers = last_net.wrappers
        last_net.wrappers = self.initialize_wrappers("", device)

        # Handle params
        self.network_params = self.NetworkParams({"wrappers": last_net.network_params.runtime["wrappers"],
                                                  "data": first_net.network_params.runtime["data"]})
        assert first_net.meta["out_channels"] == last_net.meta["in_channels"]
        self.meta = {"in_channels": first_net.meta["in_channels"], "out_channels": last_net.meta["out_channels"]}

    def __call__(self, image):
        return self.wrappers[self.stage](image, self.forward, self.model)

    def __getitem__(self, key):
        return self.networks[key]

    def forward(self, image):
        for net in self.sequence:
            image = self.networks[net](image)
        return image

    def train(self):
        for net in self.sequence:
            self.networks[net].train()
        self.stage = Network.TRAIN
        return self

    def eval(self):
        for net in self.sequence:
            self.networks[net].eval()
        self.stage = Network.EVAL
        return self

    def freeze(self, net=None):
        if net is not None:
            self.networks[net].freeze()
            return self

        for net in self.sequence:
            self.networks[net].freeze()
        self.frozen = True
        return self

    def parameters(self, optimizer_opts, net=None):
        if net is not None:
            return self.networks[net].parameters(optimizer_opts)

        acc = []
        for net in self.sequence:
            acc.append({"params": self.networks[net].parameters(optimizer_opts)})
        return acc

    @classmethod
    def initialize(cls, params, device):
        sequence = params.pop("sequence").split(",")
        for net in params:
            params[net] = NETWORKS[params[net].pop("type")].initialize(params[net], device)
        return cls(params, sequence, device=device, frozen=False)

    def overlay_params(self, new_params, device):
        if not new_params:
            return self

        diff = set(self.sequence) - set(new_params.keys())
        assert not diff, diff

        acc = {}
        for net in self.sequence:
            acc[net] = self.networks[net]
            if net in new_params:
                acc[net] = acc[net].overlay_params(new_params[net], device)
        return self.__class__(**acc, device=device, frozen=True)


    #
    # Load and save
    #

    def state_dict(self):
        network_hierarchy = {}
        state = {}
        for net in self.sequence:
            netstate = self.networks[net].state_dict()
            netstate[net] = netstate.pop("net")
            # Enforce zero overlap
            intersection = set(state.keys()).intersection(netstate.keys())
            assert not intersection, intersection
            # Nesting
            network_hierarchy[net] = [x for x in netstate if x != net]
            state.update(netstate)

        state["net"] = {
            "type": self.__class__.__name__,
            "frozen": self.frozen,
            "sequence": self.sequence,
            "network_hierarchy": network_hierarchy,
        }
        return state

    @classmethod
    def initialize_from_state(cls, state_dict, device, params, runtime):
        checkpoint = state_dict.pop("net")
        assert checkpoint["type"] == cls.__name__
        assert checkpoint.keys() == {"type", "frozen", "sequence", "network_hierarchy"}, checkpoint.keys()
        assert set(checkpoint["sequence"]) == checkpoint['network_hierarchy'].keys()

        runtime_propagated = {net: None for net in checkpoint["sequence"]}
        if runtime and "wrappers" in runtime:
            runtime_propagated[checkpoint['sequence'][-1]] = {"wrappers": runtime.pop("wrappers")}
        if runtime and "data" in runtime:
            runtime_propagated[checkpoint['sequence'][0]] = {"data": runtime.pop("data")}
        assert not runtime, runtime

        if params is not None:
            params_sequence = params["sequence"].split(",")
            assert checkpoint["sequence"] == params_sequence, \
                    "%s != %s" % (str(checkpoint["sequence"]), str(params_sequence))

        acc = {}
        for net in checkpoint["network_hierarchy"]:
            netparams = params[net] if params is not None else None
            netstate = {x: state_dict[x] for x in checkpoint["network_hierarchy"][net]}
            netstate["net"] = state_dict[net]
            acc[net] = NETWORKS[state_dict[net]["type"]].initialize_from_state(netstate, device, netparams, runtime_propagated[net])

        return cls(acc, checkpoint["sequence"], device=device, frozen=checkpoint["frozen"])


    #
    # Utils
    #

    def train_data(self):
        acc = []
        for net in self.sequence:
            train_data = self.networks[net].train_data()
            acc += [{**x, "key": x["key"].replace("net/", net+"/")} for x in train_data]
        return acc

    def const_data(self):
        acc = []
        graphs = {}
        for net in self.sequence:
            for const_data in self.networks[net].const_data():
                if const_data["key"] == "network_graph":
                    graphs[net] = const_data["data"].pop("net")
                    graphs.update(const_data["data"])
                else:
                    acc.append({**const_data, "key": "%s/%s" % (net, const_data["key"])})
        if graphs:
            acc.append({"key": "network_graph", "dtype": "blob", "data": graphs})
        return acc


    def __repr__(self):
        nice_params = "\n" + "".join("    %s: %s,\n" % (x, y) for x, y in self.network_params._asdict().items())
        nice_wrappers = "\n" + "".join("    %s: %s,\n" % (x, indent(str(y))) for x, y in self.wrappers.items())
        nice_networks = ""
        for net in self.sequence:
            nice_networks += f"        {net}: {indent(str(self.networks[net]), 2)}\n"

        return \
f"""{self.__class__.__name__} (
    sequence: {self.sequence}
    networks: {{
{nice_networks}
    }}
    meta: {self.meta}
    network_params: {{{indent(nice_params)}}}
    wrappers: {{{indent(nice_wrappers)}}}
)"""


class CirNetwork(SingleNetwork):

    def __init__(self, model, network_params, device, frozen):
        if "data" not in network_params.runtime:
            network_params.runtime["data"] = {}
        if "mean_std" not in network_params.runtime["data"]:
            network_params.runtime["data"]["mean_std"] = [model.meta["mean"], model.meta["std"]]
        super().__init__(model, network_params, device, frozen)

    @staticmethod
    def _set_batchnorm_eval(mod):
        if mod.__class__.__name__.find('BatchNorm') != -1:
            # freeze running mean and std
            mod.eval()

    def train(self):
        super().train()
        self.model.apply(CirNetwork._set_batchnorm_eval)
        return self

    def parameters(self, optimizer_opts, net="net"):
        assert net == "net"
        parameters = []
        parameters.append({'params': self.model.features.parameters()})
        if self.model.meta['local_whitening']:
            parameters.append({'params': self.model.lwhiten.parameters()})
        if not self.model.meta['regional']:
            # global, only pooling parameter p weight decay should be 0
            parameters.append({'params': self.model.pool.parameters(), 'lr': optimizer_opts['lr']*10, 'weight_decay': 0})
        else:
            # regional, pooling parameter p weight decay should be 0,
            # and we want to add regional whitening if it is there
            parameters.append({'params': self.model.pool.rpool.parameters(), 'lr': optimizer_opts['lr']*10, 'weight_decay': 0})
            if self.model.pool.whiten is not None:
                parameters.append({'params': self.model.pool.whiten.parameters()})
        # add final whitening if exists
        if self.model.whiten is not None:
            parameters.append({'params': self.model.whiten.parameters()})
        return parameters


# Initialization

NETWORKS = {
    "SingleNetwork": SingleNetwork,
    "SequentialNetwork": SequentialNetwork,
    "CirNetwork": CirNetwork,
}

def initialize_network(params, device, state=None, runtime=None):
    if params:
        network_cls = NETWORKS[params.pop("type")]
    else:
        network_cls = NETWORKS[state["net"]["type"]]

    if state:
        return network_cls.initialize_from_state(state, device, params, runtime)

    return network_cls.initialize(params, device)
