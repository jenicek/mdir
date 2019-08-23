import time
import sys
import numpy as np

from ..tools import stats

from cirtorch.utils.whiten import whitenlearn, whitenapply, pcawhitenlearn


def whiten(params, data):
    """Apply pre-computed whitening"""
    dimensions = params.pop("dimensions", None) or None
    assert not params, params.keys()
    whitening, names, values = data
    assert len(names) == len(values)
    resources = stats.ResourceUsage()

    time0 = time.time()
    whitened = whitenapply(values.T, whitening['m'], whitening['P'], dimensions)
    timing = time.time() - time0
    metadata = {"timings": {"whitening_apply": round(timing, 2)},
                "resource_usage": resources.take_current_stats().get_resources()}

    return metadata, names, whitened.T


def learn_lw_whitening(params, data):
    """Learn Lw whitening"""
    assert not params
    names, values, queries, positives = data
    assert len(names) == len(values)
    assert len(queries) == len(positives)

    # Handle data
    values = values.astype(np.float64).T
    name_index = {x: i for i, x in enumerate(names)}
    qidxs = np.array([name_index[x] for x in queries])
    pidxs = np.array([name_index[x] for x in positives])

    resources = stats.ResourceUsage()

    # Learn the whitening on a shuffled subset of the data if the matrix is not positive definite
    time0 = time.time()
    max_trials = 100
    max_excluded = 0.95
    trial = 0
    while True:
        try:
            if trial == 0:
                qwhit, pwhit = qidxs, pidxs
            else:
                idxs = np.random.permutation(len(qidxs))[:int(len(qidxs) * (1 - trial/max_trials*max_excluded))]
                print("Using subset of queries (%s/%s) trial %s" % (len(idxs), len(qidxs), trial), file=sys.stderr)
                qwhit, pwhit = qidxs[idxs], pidxs[idxs]

            whit_m, whit_p = whitenlearn(values, qwhit, pwhit)
            break
        except np.linalg.linalg.LinAlgError as e:
            if str(e) != "Matrix is not positive definite" or trial >= max_trials-1:
                raise
            trial += 1
    timing = time.time() - time0

    metadata = {"stats": {"failed_times": trial,
                          "vectors_used": round(len(qwhit) / float(len(qidxs)), 2),
                          "vectors_total": len(qidxs)},
                "timings": {"whitening_learn": round(timing, 2)},
                "resource_usage": resources.take_current_stats().get_resources()}
    return metadata, {'m': whit_m, 'P': whit_p}


def learn_pca_whitening(params, data):
    """Learn PCA whiteining"""
    shrink = params.pop("shrink", None) or None
    assert not params
    values, = data

    values = values.astype(np.float64).T

    resources = stats.ResourceUsage()
    time0 = time.time()
    whit_m, whit_P = pcawhitenlearn(values, shrink)
    timing = time.time() - time0

    metadata = {"timings": {"whitening_learn": round(timing, 2)},
                "resource_usage": resources.take_current_stats().get_resources()}
    return metadata, {'m': whit_m, 'P': whit_P}


def paste_pca_normalize(params, data):
    """Concatenate vectors horizontally and optionally apply pca dimension reduction"""
    dimensions = params.pop("dimensions") or None
    assert not params
    assert len(set(len(x) for x in data)) == 1

    if data[0].shape == (0,):
        return {}, data[0]

    value = np.concatenate(data, axis=1)

    if dimensions:
        resources = stats.ResourceUsage()
        time0 = time.time()

        value -= np.mean(value)
        eigval, eigvec = np.linalg.eig(value.T.dot(value))
        vecs = eigvec[:,np.argsort(eigval)[-dimensions:]]
        value = value.dot(vecs.dot(vecs.T))

        timing = time.time() - time0
        metadata = {"timings": {"pca_compute": round(timing, 2)},
                    "resource_usage": resources.take_current_stats().get_resources()}
    else:
        metadata = {}

    value = value / np.expand_dims(np.linalg.norm(value, axis=1), axis=1)

    return metadata, value
