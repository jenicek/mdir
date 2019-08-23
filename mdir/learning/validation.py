import copy

from ..tools.utils import get_dataset_params
from ..tools.stats import StopWatch
from ..tools.utils import indent
from ..components.optim.criterion import initialize_criterion
from ..components.optim.score import initialize_score
from ..components.data.dataset import initialize_dataset_loader


class NoValidation:

    decisive_criterion = ""

    def validations(self, _epoch):
        return []

    def __repr__(self):
        return f"""{self.__class__.__name__} ()"""


class SingleValidation:

    def __init__(self, data_loader, criterion, network_overlay, frequency):
        self.data_loader = data_loader
        self.criterion = criterion
        self.network_overlay = network_overlay
        self.frequency = frequency
        self.decisive_criterion = "val/learning/%s:total" % ("loss" if data_loader else "score")

        self.criterion_mean_reduction = None
        if self.data_loader:
            assert criterion.reduction in {"mean", "sum"}, criterion.reduction
            self.criterion_mean_reduction = criterion.reduction == "mean"

    @classmethod
    def initialize(cls, params_validation, data, params_data, default_criterion, net_defaults):
        data_key = params_validation.pop("data")
        if data_key is None:
            data_loader = None
        else:
            data_params = get_dataset_params(params_data[data_key], net_defaults)
            data_loader = initialize_dataset_loader(data, "val", copy.deepcopy(data_params))

        criterion_section = params_validation.pop("criterion")
        if criterion_section == "default":
            if default_criterion is None:
                raise ValueError("Criterion cannot be 'default' when default criterion is not specified")
            criterion = default_criterion
        elif data_loader is None:
            criterion = initialize_score(get_dataset_params(criterion_section, net_defaults))
        else:
            criterion = initialize_criterion(criterion_section)

        network_overlay = params_validation.pop("network_overlay")
        frequency = params_validation.pop("frequency")

        assert not params_validation, params_validation.keys()

        return cls(data_loader=data_loader, criterion=criterion, network_overlay=network_overlay, frequency=frequency)

    def validations(self, epoch):
        return [("val", self)] if self.should_validate(epoch) else []

    def should_validate(self, epoch):
        return epoch is None or (self.frequency and (epoch + 1) % self.frequency == 0)

    def validate(self, network, device, logger):
        network = network.overlay_params(copy.deepcopy(self.network_overlay), device)
        val_loader = self.data_loader
        stopwatch = StopWatch()

        network.eval()

        if not val_loader:
            return self.criterion(network, device, logger)

        if hasattr(val_loader.dataset, "prepare_epoch"):
            metadata = val_loader.dataset.prepare_epoch(network, device)
            stopwatch.lap("prepare_data")
            if metadata:
                logger(None, len(val_loader), "data_mining", metadata, "scalar/loss")
            logger(None, len(val_loader), "prepare_epoch", stopwatch.reset(include_total=False), "scalar/time")

        acc = []
        for i, (batch_images, batch_targets) in enumerate(val_loader):
            stopwatch.lap("prepare_data")
            if not isinstance(batch_targets, list):
                batch_targets = batch_targets.to(device)
            loss = self.criterion(network(batch_images), batch_targets).item()
            # Always report batch-normalized
            if not self.criterion_mean_reduction:
                loss /= len(batch_images)
            stopwatch.lap("process_batch")
            logger(i, len(val_loader), "loss", {"total": loss}, "scalar/loss")
            logger(i, len(val_loader), "iteration", stopwatch.reset(include_total=False), "scalar/time")
            acc.append(loss)

        return acc

    def __repr__(self):
        return \
f"""{self.__class__.__name__} (
    dataset: {indent(str(self.data_loader.dataset))}
    criterion: {indent(str(self.criterion))}
    network_overlay: {indent(str(self.network_overlay))}
    frequency: {self.frequency}
    decisive_criterion: {self.decisive_criterion}
    criterion_mean_reduction: {self.criterion_mean_reduction}
)"""


class MultiCriterialValidation:

    def __init__(self, decisive_criterion, validations):
        self.decisive_criterion = decisive_criterion
        self.vals = validations

    @classmethod
    def initialize(cls, params_validation, **kwargs):
        decisive_criterion = params_validation.pop("decisive_criterion")
        validations = {}
        for key, scenario in params_validation.items():
            validations[key] = initialize_validation(scenario, **kwargs)

        return cls(decisive_criterion, validations)

    def validations(self, epoch):
        acc = {}
        for key, val in self.vals.items():
            if val.should_validate(epoch):
                acc[key] = val
        return acc.items()

    def __repr__(self):
        nice_validations = "\n" + "".join("%s: %s\n" % (x, y) for x, y in self.vals.items())
        return \
f"""{self.__class__.__name__} (
    decisive_criterion: {self.decisive_criterion}
    validations: {indent(nice_validations)}
)"""


VALIDATIONS = {
    "SingleValidation": SingleValidation,
    "MultiCriterialValidation": MultiCriterialValidation,
}

def initialize_validation(params, **kwargs):
    if isinstance(params, bool) and not params:
        return NoValidation()

    return VALIDATIONS[params.pop("type")].initialize(params, **kwargs)
