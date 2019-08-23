import os.path
import time
import math
import pickle

import torch
from torchvision import transforms

from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.datasets.datahelpers import cid2filename
from cirtorch.utils.whiten import whitenlearn, whitenapply
from cirtorch.utils.general import get_data_root, htime

from daan.ml.tools import path_join


def embed(params, data):
    net = params.pop("net")
    imgdir = params.pop("imgdir")
    whitening = params.pop("whitening", None)
    whitening_dir = params.pop("whitening_dir", None)
    image_size = params.pop("image_size", 1024)
    multiscale = params.pop("multiscale", True)
    assert not params, params.keys()
    input_images, bbxs = (data[0], None) if len(data) == 1 else data
    impaths = [path_join(imgdir, x) for x in input_images]
    if not data[0]:
        return ({"status": "skipped"}, [], []) + (([],) if whitening_dir else tuple())

    # Handle paths
    assert os.path.exists(net), net

    # loading network from path
    print(">> Loading network:\n>>>> '{}'".format(net))
    state = torch.load(net)
    net = init_network({'architecture': state['meta']['architecture'],
                        'pooling': state['meta']['pooling'],
                        'whitening': state['meta']['whitening'],
                        'mean': state['meta']['mean'],
                        'std': state['meta']['std'],
                        'pretrained': False})
    net.load_state_dict(state['state_dict'])
    print(">>>> loaded network: ")
    print(net.meta_repr())

    # setting up the multi-scale parameters
    ms = multiscale if not isinstance(multiscale, bool) else [1, 1./math.sqrt(2), 1./2] if multiscale else [1]
    if net.meta['pooling'] == 'gem' and net.whiten is None:
        msp = net.pool.p.data.tolist()[0]

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()
    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # compute whitening
    if whitening_dir:
        whitening_dir = os.path.join(whitening_dir, "%s_%s_%s_%s.lw.pkl" % (whitening, None, image_size, multiscale))
        print('>> {}: Loading whitening...'.format(whitening))
        with open(whitening_dir, "rb") as handle:
            Lw = pickle.load(handle)
    # elif whitening:
    #     Lw, _ = _compute_whitening(whitening, net, image_size, transform, ms, msp)
    else:
        Lw = None

    # extract database and query vectors
    print('>> Images descriptors...')
    vecs = extract_vectors(net, impaths, image_size, transform, bbxs=bbxs, ms=ms, msp=msp)

    print('>> Evaluating...')

    # convert to numpy
    vecs = vecs.numpy()

    if Lw is not None:
        # whiten the vectors
        vecs_lw  = whitenapply(vecs, Lw['m'], Lw['P'])
        return {}, input_images, vecs.T, vecs_lw.T

    return {}, input_images, vecs.T


def learn_whitening(params, data):
    net = params.pop("net")
    whitening = params.pop("whitening")
    whitening_dir = params.pop("whitening_dir", None)
    image_size = params.pop("image_size", 1024)
    multiscale = params.pop("multiscale", True)
    params.pop("imgdir", None)
    assert not params
    assert not data

    # Handle paths
    assert os.path.exists(net), net

    # Handle whitening
    if whitening in ['sfm30k', 'sfm120k']:
        whitening = {
            'sfm30k': 'retrieval-SfM-30k',
            'sfm120k': 'retrieval-SfM-120k',
        }[whitening]

    # loading network from path
    print(">> Loading network:\n>>>> '{}'".format(net))
    state = torch.load(net)
    net = init_network({'architecture': state['meta']['architecture'],
                        'pooling': state['meta']['pooling'],
                        'whitening': state['meta']['whitening'],
                        'mean': state['meta']['mean'],
                        'std': state['meta']['std'],
                        'pretrained': False})
    net.load_state_dict(state['state_dict'])
    print(">>>> loaded network: ")
    print(net.meta_repr())

    # setting up the multi-scale parameters
    ms = multiscale if not isinstance(multiscale, bool) else [1, 1./math.sqrt(2), 1./2] if multiscale else [1]
    if net.meta['pooling'] == 'gem' and net.whiten is None:
        msp = net.pool.p.data.tolist()[0]

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()
    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    Lw, elapsed = _compute_whitening(whitening, net, image_size, transform, ms, msp)

    # Save
    if whitening_dir: # back-compatible option
        os.makedirs(whitening_dir, exist_ok=True)
        whitening_dir = os.path.join(whitening_dir, "%s_%s_%s_%s.lw.pkl" % (whitening, None, image_size, multiscale))
        with open(whitening_dir, "wb") as handle:
            pickle.dump(Lw, handle)
        return {"whitening_learn": int(elapsed)},

    return {"whitening_learn": int(elapsed)}, Lw


def convert_contained_net(params, data):
    source = params.pop("source")
    net = params.pop("net")
    assert not params
    assert not data

    # Handle paths
    assert os.path.exists(source), source

    # Loading whitening
    print(">> Loading network:\n>>>> '{}'".format(source))
    official = torch.load(source, map_location=lambda storage, location: storage)
    meta = official.pop("meta")
    net_state = {
        "type": "CirNetwork",
        "network_params": {
            "model": {
                "architecture": "cirnet",
                "cir_architecture": meta.pop("architecture"),
                "local_whitening": meta.pop("local_whitening", False),
                "pooling": meta.pop("pooling"),
                "regional": meta.pop("regional", False),
                "whitening": meta.pop("whitening"),
                "pretrained": True,
            },
            "runtime": {
                "wrappers": "",
                "data": {
                    "mean_std": [meta.pop("mean"), meta.pop("std")],
                    "transforms": "pil2np | totensor | normalize",
                },
            },
        },
        "model_state": official.pop("state_dict"),
    }

    # Integrity check
    del meta["outputdim"]
    del meta["Lw"]
    assert not meta, meta
    assert not official, official

    if not os.path.exists(os.path.dirname(net)):
        os.makedirs(os.path.dirname(net))
    torch.save(net_state, net)
    return {},


def load_whitening(params, data):
    net = params.pop("net")
    whitening = params.pop("whitening")
    whitening_dir = params.pop("whitening_dir", None)
    image_size = params.pop("image_size", 1024)
    multiscale = params.pop("multiscale", True)
    params.pop("imgdir", None)
    assert not params
    assert not data

    # Handle paths
    assert os.path.exists(net), net

    # Handle whitening
    if whitening in ['sfm30k', 'sfm120k']:
        whitening = {
            'sfm30k': 'retrieval-SfM-30k',
            'sfm120k': 'retrieval-SfM-120k',
        }[whitening]

    # Loading whitening
    print(">> Loading network:\n>>>> '{}'".format(net))
    state = torch.load(net, map_location=lambda storage, location: storage)
    assert isinstance(multiscale, bool)
    Lw = state['meta']['Lw'][whitening]["ms" if multiscale else "ss"]

    if whitening_dir: # compatibility mode
        # Save
        os.makedirs(whitening_dir, exist_ok=True)
        whitening_dir = os.path.join(whitening_dir, "%s_%s_%s_%s.lw.pkl" % (whitening, None, image_size, multiscale))
        with open(whitening_dir, "wb") as handle:
            pickle.dump(Lw, handle)
        return {},

    return {}, Lw


def _compute_whitening(whitening, net, image_size, transform, ms, msp):
    # compute whitening
    start = time.time()

    print('>> {}: Learning whitening...'.format(whitening))

    # loading db
    db_root = os.path.join(get_data_root(), 'train', whitening)
    ims_root = os.path.join(db_root, 'ims')
    db_fn = os.path.join(db_root, '{}-whiten.pkl'.format(whitening))
    with open(db_fn, 'rb') as f:
        db = pickle.load(f)
    images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]

    # extract whitening vectors
    print('>> {}: Extracting...'.format(whitening))
    wvecs = extract_vectors(net, images, image_size, transform, ms=ms, msp=msp)

    # learning whitening
    print('>> {}: Learning...'.format(whitening))
    wvecs = wvecs.numpy()
    m, P = whitenlearn(wvecs, db['qidxs'], db['pidxs'])
    Lw = {'m': m, 'P': P}

    elapsed = time.time()-start
    print('>> {}: elapsed time: {}'.format(whitening, htime(elapsed)))

    return Lw, elapsed
