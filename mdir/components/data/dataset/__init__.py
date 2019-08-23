from torch.utils.data import DataLoader
from . import cirtorch_datasets, tuple_datasets
from ..transform import initialize_transforms


# Initialization

DATASET_LABELS = {
    "RandomImageTuple": tuple_datasets.RandomImageTupleDataset,
    "PregeneratedImageTuple": tuple_datasets.PregeneratedImageTupleDataset,
    "CirTuples": cirtorch_datasets.cir_tuples_dataset,
    "CirImageList": cirtorch_datasets.cir_image_list_dataset,
}

LOADER_DEFAULT_PARAMS = {
    "shuffle": False,
    "num_workers": 6,
    "pin_memory": True,
}

def initialize_dataset(data, stage, transform, params):
    if stage == "train" or stage == "val":
        if data:
            col_start, col_end = params.pop("data_cols").split(":")
            data = data[int(col_start):(int(col_end) if col_end else None)]
    elif stage != "test":
        raise RuntimeError("Unsupported stage '%s'" % stage)

    return DATASET_LABELS[params.pop("name")](data, transform=transform, **params)

def initialize_dataset_loader(data, stage, params, loader_default_params=None):
    transform = initialize_transforms(params.pop("transforms"), mean_std=params.pop("mean_std"))
    dataset = initialize_dataset(data, stage, transform, params.pop("dataset"))
    loader_params = {**LOADER_DEFAULT_PARAMS, **(loader_default_params or {}), **dataset.loader_params, **params.pop("loader", {})}
    assert "batch_size" in loader_params
    assert not params, params.keys()
    return DataLoader(dataset, **loader_params)
