import abc
import os.path
import numpy as np
from concurrent import futures
from PIL import Image

from daan.ml.tools import path_join
from ...tools import imgtools

THREAD_WORKERS = 6


class DataOutput(abc.ABC):

    @abc.abstractmethod
    def preprocess(self):
        """Open the data output giving it the total number of data that should be processed"""

    @abc.abstractmethod
    def add(self, index, input_data, output_data):
        """Add a single data output"""

    @abc.abstractmethod
    def postprocess(self):
        """Should be called after the last data is processed"""


class RgbImageSaver(DataOutput):

    def __init__(self, data, data_params, *, image_dir, dir_structure=None, append=False,
                 stretch_by=False):
        # Data
        assert len(data) == 1
        data = data[0]
        self.data = [x if isinstance(x, (list, tuple)) else [x] for x in data]

        # Params
        self.dataset = {
            "mean_std": data_params["mean_std"],
            "transforms": data_params["transforms"],
        }
        self.image_dir = image_dir
        if dir_structure is None:
            dir_structure = "flat" if len(self.data[0]) > 1 else "input"
        self.dir_structure = dir_structure
        self.append = append
        self.stretch_by = stretch_by

        # Runtime
        self.fnames = None
        self.paths = None

    def preprocess(self):
        # Build filenames
        if self.dir_structure == "flat":
            fnames = []
            for item in self.data:
                fname_pieces = [x.rsplit(".", 1)[0] for x in item[:-1]] + [item[-1]]
                fnames.append("::".join(fname_pieces).replace("/", "%"))
        else:
            fnames = [x[0] for x in self.data]

        # Build paths and exclude already performed
        paths = [path_join(self.image_dir, x) for x in fnames]
        if self.append:
            idxs = [i for i, x in enumerate(paths) if not os.path.exists(x)]
            data = [fnames[i] for i in idxs]
            paths = [paths[i] for i in idxs]

        self.fnames = fnames
        self.paths = paths
        return data,

    def add(self, index, input_data, output_data):
        """Save image in RGB"""
        # Convert
        img = imgtools.get_image((input_data[0].cpu(), output_data[0].cpu()),
                                 self.dataset['mean_std'],
                                 self.dataset['transforms'],
                                 stretch_by=self.stretch_by)
        # Store
        os.makedirs(os.path.dirname(self.paths[index]), exist_ok=True)
        Image.fromarray(img).save(self.paths[index])

    def postprocess(self):
        return self.fnames,


class AsyncOutput(DataOutput):

    def __init__(self, output):
        self.output = output
        self.pool = None
        self.buf = None

    def preprocess(self):
        self.pool = futures.ThreadPoolExecutor(max_workers=THREAD_WORKERS)
        self.buf = []
        return self.output.preprocess()

    def add(self, index, input_data, output_data):
        result = self.pool.submit(self.output.add, index, input_data.cpu(), output_data.cpu())
        if len(self.buf) >= THREAD_WORKERS*2:
            self.buf.pop(0).result()
        self.buf.append(result)

    def postprocess(self):
        # Shutdown async saver
        for item in self.buf:
            item.result()
        self.pool.shutdown(wait=True)

        # Return nested output
        return self.output.postprocess()


class EmbeddingOutput(DataOutput):

    def __init__(self, data, _data_params, *, bbxs=False):
        if not bbxs:
            assert len(data) == 1, len(data)
        self.images, self.bbxs = data if bbxs else (data[0], None)
        self.vecs = None

    def preprocess(self):
        return self.images, self.bbxs

    def add(self, index, input_data, output_data):
        if input_data is None and output_data is None:
            self.vecs[index,:] = np.nan
            return

        vec = output_data.cpu().numpy()
        if self.vecs is None:
            self.vecs = np.zeros((len(self.images), vec.shape[0]))
        self.vecs[index,:] = vec

    def postprocess(self):
        return self.images, self.vecs if self.vecs is not None else []


OUTPUT_LABELS = {
    "embedding": EmbeddingOutput,
    "rgb": RgbImageSaver,
}


# Init function

def initialize_output(output, data_params, data):
    async_param = output.pop("async", False)
    output = OUTPUT_LABELS[output.pop("name")](data, data_params, **output)
    if async_param:
        output = AsyncOutput(output)
    return output
