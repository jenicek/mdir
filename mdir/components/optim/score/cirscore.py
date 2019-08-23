import os.path
import numpy as np

from daan.ml.tools import path_join
from daan.data.file_readers import initialize_file_reader

from ...data.transform import initialize_transforms
from ....tools.stats import StopWatch

from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.general import get_data_root
from cirtorch.utils.evaluate import compute_map_and_print
from cirtorch.networks.imageretrievalnet import extract_vectors


class CirDatasetAp:

    def __init__(self, params):
        # prepare config structure for the test dataset
        self.image_size = params.pop("image_size")
        self.dataset = params.pop("dataset")
        self.transforms = initialize_transforms(params.pop("transforms"), params.pop("mean_std"))

        if isinstance(self.dataset, dict):
            # Tsv dataset files
            assert self.dataset.keys() == {"name", "queries", "db", "imgdir"}
            imgdir = self.dataset['imgdir']
            with initialize_file_reader(self.dataset['db'], keys=["identifier"]) as reader:
                data = reader.get()
                self.images = [path_join(imgdir, x) for x in data["identifier"]]
                mapping = {x: i for i, x in enumerate(data["identifier"])}
            with initialize_file_reader(self.dataset['queries'], keys=["query", "bbx", "ok", "junk"]) as reader:
                data = reader.get()
                self.qimages = [path_join(imgdir, x) for x in data["query"]]
                self.bbxs = [tuple(x) if x else None for x in data["bbx"]]
                self.gnd = [{'ok': [mapping[x] for x in ok], 'junk': [mapping[x] for x in junk]} \
                                for ok, junk in zip(data["ok"], data["junk"])]
            self.dataset = self.dataset['name']
        else:
            # Official cirtorch files
            cfg = configdataset(self.dataset, os.path.join(get_data_root(), 'test'))
            self.images = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
            self.qimages = [cfg['qim_fname'](cfg,i) for i in range(cfg['nq'])]
            self.bbxs = [tuple(cfg['gnd'][i]['bbx']) if cfg['gnd'][i]['bbx'] else None for i in range(cfg['nq'])]
            self.gnd = cfg['gnd']

        assert not params, params.keys()

    def __call__(self, network, device, logger):
        stopwatch = StopWatch()

        # extract database and query vectors
        print('>> {}: database images...'.format(self.dataset))
        vecs = extract_vectors(network, self.images, self.image_size, self.transforms, device=device)
        print('>> {}: query images...'.format(self.dataset))
        if self.images == self.qimages and set(self.bbxs) == {None}:
            qvecs = vecs.clone()
        else:
            qvecs = extract_vectors(network, self.qimages, self.image_size, self.transforms, device=device, bbxs=self.bbxs)
        stopwatch.lap("extract_descriptors")

        print('>> {}: Evaluating...'.format(self.dataset))

        # convert to numpy
        vecs = vecs.numpy()
        qvecs = qvecs.numpy()

        # search, rank, and print
        scores = np.dot(vecs.T, qvecs)
        ranks = np.argsort(-scores, axis=0)
        averages, scores = compute_map_and_print(self.dataset, ranks, self.gnd)
        stopwatch.lap("compute_score")

        first_score = scores[list(scores.keys())[0]]
        logger(None, len(first_score), "dataset", stopwatch.reset(), "scalar/time")
        logger(None, len(first_score), "score_avg", averages, "scalar/score")

        assert len({len(x) for x in scores.values()}) == 1
        for i, _ in enumerate(first_score):
            logger(i, len(first_score), "score", {x: scores[x][i] for x in scores}, "scalar/score")
