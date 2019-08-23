import copy
from collections import namedtuple
from .checkpoints import Checkpoints
from .network import initialize_network
from .training import  initialize_training
from .validation import initialize_validation
from ..tools.eventprocessor import initialize_processor
from ..tools.utils import indent
from ..tools.stats import ResourceUsage, CodeVersion


class TrainValLearning:

    Epoch = namedtuple('Epoch', ['epoch', 'train', 'vals'])

    def __init__(self, params, network, training, validation, events, resources, checkpoints):
        self.params = params
        self.network = network
        self.training = training
        self.validation = validation
        self.events = events
        self.resources = resources
        self.checkpoints = checkpoints
        self.code_version = CodeVersion()

    @classmethod
    def initialize(cls, params, data, device):
        store_params = copy.deepcopy(params)
        assert params.keys() == {"network", "learning", "output", "data"}, params.keys()
        assert params["learning"]["type"] == cls.__name__, params["learning"]["type"]
        assert params["learning"].keys() == {"type", "checkpoints", "training", "validation"}, params["learning"].keys()

        checkpoints = Checkpoints(**params["learning"]["checkpoints"])
        state = checkpoints.load_latest_epoch(params['learning']['training']['epochs'])
        if state is not None:
            network = initialize_network(params["network"], device, state[0], None)
            training = initialize_training(params["learning"]["training"], network, data, params["data"], device, state[1]["training"])
            events = initialize_processor(params["output"]["learning"], checkpoints.directory / "../epochs", state[1]["events"])
            resources = ResourceUsage.initialize_from_state(state[1]["resources"])
        else:
            network = initialize_network(params["network"], device)
            training = initialize_training(params["learning"]["training"], network, data, params["data"], device)
            events = initialize_processor(params["output"]["learning"], checkpoints.directory / "../epochs")
            resources = ResourceUsage.initialize()

        if state is not None:
            assert state[1]["validation"]["params"] == params["learning"]["validation"], \
                    "%s != %s" % (str(state[1]["validation"]["params"]), str(params["learning"]["validation"]))
            assert state[1]["datasets"] == params["data"], \
                    "%s != %s" % (str(state[1]["datasets"]), str(params["data"]))

        # Validation init
        net_defaults = network.network_params.runtime.get("data", {})
        validation = initialize_validation(params["learning"]["validation"], data=data, params_data=params["data"],
                                           default_criterion=training.criterion, net_defaults=net_defaults)

        return cls(store_params, network, training, validation, events, resources, checkpoints)

    def close_epoch(self):
        self.events.close_epoch()
        train_stats = {
            "training": self.training.state_dict(),
            "validation": { "params": self.params["learning"]["validation"] },
            "datasets": self.params["data"],
            "events": self.events.state_dict(),
            "resources": self.resources.state_dict(),
        }
        self.checkpoints.save_epoch(self.network.state_dict(), train_stats, self.training.epoch,
                                    self.events.metadata.is_last_best(self.validation.decisive_criterion),
                                    not self.training.remains_epochs)

    @property
    def metadata(self):
        return {
            "metrics": self.events.metadata.metadata(),
            "best_epoch": self.events.metadata.best_epoch(self.validation.decisive_criterion),
            "resource_usage": self.resources.get_resources(),
            "code_version": self.code_version.versions,
        }

    def __iter__(self):
        return self

    def __next__(self):
        epoch, train = next(self.training)
        return self.Epoch(epoch=epoch, train=train, vals=self.validation.validations(epoch))

    def __repr__(self):
        return \
f"""{self.__class__.__name__} (
    network: {{{indent(str(self.network))}}}
    training: {{{indent(str(self.training))}}}
    validation: {{{indent(str(self.validation))}}}
)"""


LEARNINGS = {
    "TrainValLearning": TrainValLearning,
}
