from .base_optimizers import initialize_base_optimizer
from ....tools.utils import indent


class OptimizerAlternation:
    def __init__(self, optimizers, alternate_iteration, order):
        # alternate_iteration:
        # -1 = no alternation, only one optimizer active
        # 0 = no alternation, all optimizers active
        # positive n = alternate after n steps
        if len(optimizers) == 1:
            assert alternate_iteration is None
            self.names = list(optimizers.keys())
            self.optimizers = list(optimizers.values())
        else:
            assert alternate_iteration is not None
            order = order.split(",")
            assert optimizers.keys() == set(order)
            self.names = order
            self.optimizers = [optimizers[x] for x in order]

        self.alternate_iteration = alternate_iteration
        self.current_iteration = 0
        self.current_optimizer = 0

    def __iter__(self):
        return iter(self.names)

    def __getitem__(self, key):
        return self.optimizers[self.names.index(key)]

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self):
        self.current_iteration += 1
        if self.alternate_iteration:
            self.optimizers[self.current_optimizer].step()
            if self.current_iteration % self.alternate_iteration == 0:
                self.current_optimizer = (self.current_optimizer + 1) % len(self.optimizers)
        else:
            for optimizer in self.optimizers:
                optimizer.step()

    def state_dict(self):
        dct = {x: y.state_dict() for x, y in zip (self.names, self.optimizers)}
        dct["alternation"] = {
            "iteration": self.current_iteration,
            "optimizer": self.current_optimizer,
        }
        return dct

    def load_state_dict(self, state_dict):
        self.current_iteration = state_dict["alternation"].pop("iteration")
        self.current_optimizer = state_dict["alternation"].pop("optimizer")
        assert not state_dict.pop("alternation")

        assert state_dict.keys() == set(self.names)
        for name, opt in zip(self.names, self.optimizers):
            opt.load_state_dict(state_dict[name])

    @classmethod
    def initialize(cls, network, optimizers, **params):
        acc = {}
        for net in list(optimizers.keys()):
            if optimizers[net] is not None:
                acc[net] = initialize_base_optimizer(network.parameters(optimizers[net], net), optimizers[net])
            else:
                network.freeze(net)
        return cls(acc, **params)

    def __repr__(self):
        return \
f"""{self.__class__.__name__} (
    names: {self.names}
    optimizers: {indent(str(self.optimizers), 2)}
    alternation: {{
        alternate_iteration: {self.alternate_iteration}
        current_iteration: {self.current_iteration}
        current_optimizer: {self.current_optimizer}
    }}
)"""


# Initialization

OPTIMiZER_COMPOSITIONS = {
    "alternation": OptimizerAlternation,
}

def initialize_optimizer_composition(network, params):
    return OPTIMiZER_COMPOSITIONS[params["composition"].pop("type")].initialize(network=network, optimizers=params, **params.pop("composition"))
