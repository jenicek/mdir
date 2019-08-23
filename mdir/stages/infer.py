import os
import copy
import torch
import numpy as np

from ..tools.utils import get_dataset_params
from ..tools import stats
from ..components.data.dataset import initialize_dataset_loader
from ..components.data.output import initialize_output
from ..learning import load_network

# Limit threads
torch.set_num_threads(3)
os.environ['MKL_NUM_THREADS'] = "3"
os.environ['OMP_NUM_THREADS'] = "3"


def infer(params, data):
    # General initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(0)
    torch.manual_seed(0)

    if not data[0]:
        # Speedup nothing-done scenario by not loading the cnn on gpu - can be removed without consequences
        output_tmp = initialize_output(copy.deepcopy(params["output"]["inference"]), get_dataset_params(params['data']['test'], {}), data)
        if not output_tmp.preprocess()[0]:
            return ({"status": "skipped"},) + output_tmp.postprocess()

    # Load network and dataset
    network = load_network(params["network"], device).eval()
    data_params = get_dataset_params(params['data']['test'], network.network_params.runtime.get("data", {}))

    # Output init
    output = initialize_output(copy.deepcopy(params["output"]["inference"]), copy.deepcopy(data_params), data)
    data = output.preprocess()
    if not data[0]:
        return ({"status": "skipped"},) + output.postprocess()

    # Data init
    loader = initialize_dataset_loader(data, "test", copy.deepcopy(data_params), {"batch_size": 1})

    # Stats
    meter = stats.AverageMeter("Infer", len(loader), debug=params["output"].get("debug", False))
    resources = stats.ResourceUsage()

    # Get descriptors
    with torch.no_grad():
        for i, indata in enumerate(loader):
            if isinstance(indata, dict) and indata == {}:
                output.add(i, None, None)
            else:
                output.add(i, indata, network(indata))

            # Stats
            if i == len(loader)-1:
                resources.take_current_stats()
            meter.update(i, None)

    # vecs = extract_vectors(network, data[0], testing["image_size"], val_transforms, bbxs=bbxs,
    #                        ms=ms, msp=msp, device=device)
    metadata = {"stats": meter.total_stats(),
                "resource_usage": resources.get_resources()}
    return (metadata,) + output.postprocess()
