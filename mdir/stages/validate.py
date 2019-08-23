import os
import numpy as np
import torch

from ..learning import load_network
from ..learning.validation import initialize_validation
from ..tools.eventprocessor import initialize_processor

# Limit threads
torch.set_num_threads(3)
os.environ['MKL_NUM_THREADS'] = "3"
os.environ['OMP_NUM_THREADS'] = "3"


def validate(params, data):
    # General initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(0)
    torch.manual_seed(0)

    # Load network and validation
    assert params.keys() == {"network", "validation", "data"}, params.keys()
    network = load_network(params["network"], device).eval()
    net_defaults = network.network_params.runtime.get("data", {})
    validation = initialize_validation(params["validation"], data=data, params_data=params["data"],
                                       default_criterion=None, net_defaults=net_defaults)

    # Initialize logging
    events = initialize_processor({"progress": {"print_each": 100, "key_suffix": "validation/loss:total"}}, dataroot=None)

    # Validation
    with torch.no_grad():
        for val, valtask in validation.validations(epoch=None):
            logger = lambda iteration, size, label, value, dtype: \
                events.register_data(0, iteration, size, "%s/validation/%s" % (val, label), value, dtype)
            valtask.validate(network, device, logger)

    events.close_epoch()

    return {"eval": {x: y[0] for x, y in events.metadata.metadata().items()}},
