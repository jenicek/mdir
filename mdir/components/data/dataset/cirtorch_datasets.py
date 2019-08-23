from daan.ml.tools import path_join
from cirtorch import datasets as cirdatasets


def cir_tuples_dataset(data, transform, **params):
    assert not data
    dparams = {
        "name": params.pop("dataset"),
        "mode": params.pop("split"),
        "imsize": params.pop("image_size"),
        "nnum": params.pop("neg_num"),
        "transform": transform,
        "dataset_pkl": params.pop("dataset_pkl"),
        "ims_root": params.pop("image_dir"),
    }
    dparams["qsize"] = params.pop("query_size")
    dparams["poolsize"] = params.pop("pool_size")
    assert not params, params.keys()

    dataset = cirdatasets.traindataset.TuplesDataset(**dparams)
    setattr(dataset, "loader_params", {"drop_last": True, "collate_fn": cirdatasets.datahelpers.collate_tuples})
    setattr(dataset, "prepare_epoch", dataset.create_epoch_tuples)
    return dataset


def cir_image_list_dataset(data, transform, **params):
    images, bbxs = (data[0], None) if len(data) == 1 else data
    image_dir = params.pop("image_dir")
    dataset = cirdatasets.genericdataset.ImagesFromList(
        root='',
        images=[path_join(image_dir, x) for x in images],
        imsize=params.pop("image_size"),
        bbxs=bbxs,
        transform=transform,
        **params
    )
    setattr(dataset, "loader_params", {})
    return dataset
