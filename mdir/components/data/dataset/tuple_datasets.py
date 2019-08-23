import random
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from daan.ml.tools import path_join
from daan.data.file_readers import initialize_file_reader


ImageFile.LOAD_TRUNCATED_IMAGES = True

def imread(path):
    with open(path, 'rb') as f:
        return Image.open(f).convert('RGB')


class RandomImageTupleDataset(Dataset):

    loader_params = {}

    def __init__(self, data, transform, dataset, data_key, image_dir, idx):
        assert not data
        with initialize_file_reader(dataset, keys=[data_key]) as reader:
            image_list = reader.get()[data_key]

        self.image_list = [[path_join(image_dir, y) for y in x] for x in image_list]
        self.transform = transform
        if isinstance(idx, str):
            idx = [x if x in {"any", "different"} else int(x) for x in idx.split("_")]
        self.idx = idx

        self.epoch_images = None

    @staticmethod
    def get_idx(idx, length, previous_idxs, rand):
        if idx == "any":
            return rand(length)
        elif idx == "different":
            idxs = [x for x in range(length) if x not in previous_idxs]
            return idxs[rand(len(idxs))]
        elif isinstance(idx, (list, tuple)):
            for idxi in idx:
                if idxi is not None:
                    if idxi < 0:
                        idxi = length + idxi
                    assert idxi >= 0 and idxi < length

            return rand(idx[0] or 0, idx[1] or length)

        if idx < 0:
            idx = length + idx
        assert idx >= 0 and idx < length
        return idx

    def _generate_epoch_images(self, rand):
        self.epoch_images = []
        for possible in self.image_list:
            idxs = []
            for i in self.idx:
                idxs.append(self.get_idx(i, len(possible), idxs, rand))
            self.epoch_images.append([possible[i] for i in idxs])

    def prepare_epoch(self, network, device):
        self._generate_epoch_images(np.random.randint)
        return None

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        images = [imread(x) for x in self.epoch_images[idx]]

        if self.transform:
            images = self.transform(*images)

        return images


class PregeneratedImageTupleDataset(RandomImageTupleDataset):

    def __init__(self, data, transform, dataset, data_key, image_dir, idx):
        super().__init__(data, transform, dataset, data_key, image_dir, idx)

        # Pregenerate idx with fixed random seed because of recovery from checkpoint
        rand = random.Random(0).randrange
        self._generate_epoch_images(rand)

    def prepare_epoch(self, network, device):
        return None
