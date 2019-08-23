import copy
import torch

from ..tools.utils import get_dataset_params
from ..tools.stats import StopWatch
from ..tools.utils import indent
from ..components.optim.criterion import initialize_criterion
from ..components.data.dataset import initialize_dataset_loader


class SupervisedEpoch:

    LOG_TRAINDATA_SAMPLE_EVERY = 5

    def __init__(self, data_loader, criterion, mean_std, *, batch_average, fakebatch):
        self.data_loader = data_loader
        self.criterion = criterion
        self.mean_std = mean_std
        self.epoch = None
        self.batch_average = batch_average
        self.fakebatch = fakebatch

        assert criterion.reduction in {"mean", "sum"}, criterion.reduction
        self.criterion_mean_reduction = criterion.reduction == "mean"

    @classmethod
    def initialize(cls, params_epoch, data, params_data, default_criterion, net_defaults):
        data_key = params_epoch.pop("data")
        data_params = get_dataset_params(params_data[data_key], net_defaults)
        data_loader = initialize_dataset_loader(data, "train", copy.deepcopy(data_params), {"shuffle": True})

        criterion_section = params_epoch.pop("criterion")
        if criterion_section == "default":
            if default_criterion is None:
                raise ValueError("Criterion cannot be 'default' when default criterion is not specified")
            criterion = default_criterion
        else:
            criterion = initialize_criterion(criterion_section)

        return cls(data_loader=data_loader, criterion=criterion, mean_std=data_params["mean_std"], **params_epoch)

    def steps(self, epoch):
        self.epoch = epoch
        return self

    def _optimization_step(self, network, optimizer, device, batch_images, batch_targets):
        """Perform a single optimization step. Meant to be overriden by children classes."""

        optimizer.zero_grad()

        if self.fakebatch:
            # Save gpu memory by backprop after each image
            cumloss = 0
            batch_size = len(batch_images)
            image, target = None, None
            for image, target in zip(batch_images, batch_targets):
                output = network(image)
                loss = self.criterion(output, target.to(device))

                # Handle batch average
                if self.batch_average > self.criterion_mean_reduction: # already_mean=False, batch_average=True
                    loss /= batch_size
                elif self.batch_average < self.criterion_mean_reduction: # already_mean=True, batch_average=False
                    loss *= batch_size

                loss.backward()
                cumloss += loss.item()

            # Single step for a batch
            optimizer.step()

            # Report averaged
            if not self.batch_average:
                cumloss /= batch_size
            return {"total": cumloss}, image, output, target

        # Regular step
        batch_output = network(batch_images)
        loss = self.criterion(batch_output, batch_targets.to(device))

        # Handle batch average
        if self.batch_average > self.criterion_mean_reduction: # already_mean=False, batch_average=True
            loss /= len(batch_images)
        elif self.batch_average < self.criterion_mean_reduction: # already_mean=True, batch_average=False
            loss *= len(batch_images)

        loss.backward()
        optimizer.step()

        # Report averaged
        cumloss = loss.item()
        if self.batch_average is not None and not self.batch_average:
            cumloss /= len(batch_images)
        return {"total": cumloss}, batch_images[-1], batch_output[-1], batch_targets[-1]

    def _log_parameter_weights(self, network, logger):
        with torch.no_grad():
            for train_data in network.train_data():
                logger(train_data["key"], train_data["data"], train_data["dtype"])

    def _log_traindata_sample(self, image, logger, label):
        with torch.no_grad():
            if not isinstance(image, list):
                image = [image]
            mean, std = torch.Tensor(self.mean_std[0]), torch.Tensor(self.mean_std[1])
            dbg_data = {}
            for j, img in enumerate(image):
                if len(img.size()) == 4:
                    img = img[0]
                nchans = img.size(0)
                if nchans >= 3:
                    dbg_data["image%s.rgb" % j] = {"dtype": "image:rgb", "data": img[:3].detach() * std[:3,None,None] + mean[:3,None,None]}
                    # Skip other channels
                    if j >= 3:
                        continue
                for k in range(3 if nchans >= 3 else 0, nchans):
                    dbg_data["image%s.chan%s" % (j, k+1)] = {"dtype": "image:gray", "data": img[k] * std[k,None,None] + mean[k,None,None]}
                    # Skip other channels
                    if j >= 3:
                        break

            logger("data/%s" % label, dbg_data, "blob")

    def iterate(self, network, optimizer, device, logger):
        train_loader = self.data_loader
        stopwatch = StopWatch()

        network.eval()

        if hasattr(train_loader.dataset, "prepare_epoch"):
            metadata = train_loader.dataset.prepare_epoch(network, device)
            stopwatch.lap("prepare_data")
            if metadata:
                logger(None, len(train_loader), "learning/data_mining", metadata, "scalar/loss")
            logger(None, len(train_loader), "learning/prepare_epoch", stopwatch.reset(include_total=False), "scalar/time")

        if self.epoch == 0:
            self._log_parameter_weights(network, logger=lambda *x: logger(-1, len(train_loader), *x))

        network.train()

        for i, (batch_images, batch_targets) in enumerate(train_loader):
            stopwatch.lap("prepare_data")
            step_data = self._optimization_step(network, optimizer, device, batch_images, batch_targets)
            stopwatch.lap("process_batch")
            logger(i, len(train_loader), "learning/loss", step_data[0], "scalar/loss")

            # Take stats
            if i == len(train_loader)-1:
                self._log_parameter_weights(network, logger=lambda *x: logger(i, len(train_loader), *x))
            if (i == len(train_loader)-1 and (self.epoch+1) % self.LOG_TRAINDATA_SAMPLE_EVERY == 0) \
                    or (i == 0 and self.epoch == 0):
                _losses, image, output, target = step_data
                loggeri = lambda *x: logger(i, len(train_loader), *x)
                self._log_traindata_sample(image, loggeri, "input")
                if not isinstance(image, list) and len(output.size()) == len(image.size()):
                    self._log_traindata_sample(output.cpu(), loggeri, "output")
                    self._log_traindata_sample(target, loggeri, "target")

            yield step_data

            stopwatch.lap("take_statistics")
            logger(i, len(train_loader), "learning/iteration", stopwatch.reset(include_total=False), "scalar/time")

    def __repr__(self):
        return \
f"""{self.__class__.__name__} (
    dataset: {indent(str(self.data_loader.dataset))}
    criterion: {indent(str(self.criterion))}
    fakebatch: {self.fakebatch}
    batch_average: {self.batch_average}
    criterion_mean_reduction: {self.criterion_mean_reduction}
)"""


EPOCH_ITERATIONS = {
    "SupervisedEpoch": SupervisedEpoch,
}

def initialize_epoch_iteration(params, **kwargs):
    return EPOCH_ITERATIONS[params.pop("type")].initialize(params, **kwargs)
