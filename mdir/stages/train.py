import os
import torch

from ..learning import initialize_learning

# Limit threads
torch.set_num_threads(3)
os.environ['MKL_NUM_THREADS'] = "3"
os.environ['OMP_NUM_THREADS'] = "3"


def train(params, data):
    # Initialize learning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning = initialize_learning(params, data, device)

    # Save offtheshelf versions of network only
    if learning.training.epoch == -1 and not learning.training.remains_epochs:
        learning.checkpoints.save_notrain(learning.network.state_dict())
        return {},

    # Pre-learning stats
    if learning.training.epoch == -1:
        for const_data in learning.network.const_data():
            learning.events.register_data(None, None, None, "net/%s" % const_data["key"], const_data["data"], const_data["dtype"])

    for epoch in learning:
        # Training
        logger = lambda iteration, size, label, value, dtype: \
            learning.events.register_data(epoch.epoch, iteration, size, "train/%s" % label, value, dtype)
        iterations = epoch.train.iterate(learning.network, learning.training.optimizer, device, logger)
        for i, (_losses, _input, _output, _target) in enumerate(iterations):
            # Handle resources
            if not learning.training.remains_epochs and i == len(epoch.train.data_loader)-1:
                learning.resources.take_current_stats()

        # Validation
        with torch.no_grad():
            for val, valtask in epoch.vals:
                logger = lambda iteration, size, label, value, dtype: \
                    learning.events.register_data(epoch.epoch, iteration, size, "%s/learning/%s" % (val, label), value, dtype)
                valtask.validate(learning.network, device, logger)

        learning.close_epoch()

    return learning.metadata,
