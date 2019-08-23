"""
Classes for high-level reading of files
"""

import time
import abc
import re
import json
import gzip
import lzma
from collections import OrderedDict
import pickle
import numpy as np
import h5py


class InputSelector: # pylint: disable=too-few-public-methods
    """Input selector"""

    def __init__(self, *, slice=0, partitions=1, limit=None, keys=None): # pylint: disable=redefined-builtin
        """Store params with default values"""
        assert slice < partitions
        self.slice = slice
        self.partitions = partitions
        self.limit = limit
        self.keys = keys

    @property
    def slicing(self):
        """Return array slicing object for array indexation"""
        return slice(self.slice, self.limit, self.partitions)


class GenericHandler(metaclass=abc.ABCMeta):
    """Generic file reader/writer"""

    @abc.abstractmethod
    def open(self):
        """Open file for reading/writing"""

    @abc.abstractmethod
    def close(self):
        """Close file"""

    def __enter__(self):
        """Call open()"""
        self.open()
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        """Call close()"""
        self.close()


class GenericReader(GenericHandler, metaclass=abc.ABCMeta):
    """Generic file reader"""

    retry = 3

    def __init__(self, path, selector):
        """Store path and selector"""
        self.path = path
        self.handle = None
        self.selector = selector

    @abc.abstractmethod
    def get(self):
        """Get all data at once"""

    def open(self):
        """Open the file in a robust way, retrying multiple times"""
        for i in range(self.retry+1):
            try:
                return self._open()
            except (FileNotFoundError, OSError, EOFError):
                if i == self.retry:
                    raise ValueError("Error with path '%s' (try %s)" % (self.path, i+1))
                time.sleep(np.power(i+1, 3))

    def _open(self):
        """Perform the file opening, called from open()"""
        fopen = lzma.open if self.path.endswith(".xz") else gzip.open if self.path.endswith(".gz") \
                else open
        self.handle = fopen(self.path, "rb")

    def close(self):
        self.handle.close()

    @staticmethod
    def str2collection(value):
        """Convert string to a collection by calling json.load() if in json format"""
        if isinstance(value, str) and not value:
            return None
        elif isinstance(value, str) and value and ((value[0], value[-1]) == ("[", "]") or \
                (value[0], value[-1]) == ("{", "}")):
            return json.loads(value)
        return value


class TsvReader(GenericReader):
    """Read .tsv file in tsv format"""

    VALID_SUFFIXES = [".tsv", ".tsv.gz", ".tsv.xz", ".csv", ".csv.gz", ".csv.xz"]

    def __init__(self, path, selector):
        """Store path and selector"""
        super().__init__(path, selector)
        assert sum(path.endswith(x) for x in self.VALID_SUFFIXES)
        self.header = None
        self.separator = "\t" if "tsv" in path.rsplit(".", 2) else ","

    def open(self):
        super().open()
        self.header = next(self.handle).decode("utf8").strip().split(self.separator)

    def get(self):
        indexes = list(range(len(self.header)))
        if self.selector.keys:
            indexes = [self.header.index(x) for x in self.selector.keys]

        acc = [[] for _ in indexes]
        index = 0
        for line in self.handle:
            if index % self.selector.partitions == self.selector.slice:
                line = line.decode("utf8").strip("\n").split(self.separator)
                for i, j in enumerate(indexes):
                    acc[i].append(self.str2collection(line[j]))

            index += 1
            if self.selector.limit and index >= self.selector.limit:
                break

        return OrderedDict(zip([self.header[i] for i in indexes], acc))


class PklReader(GenericReader):
    """Read .pkl file in pickle format"""

    def __init__(self, path, selector):
        """Store path and selector"""
        super().__init__(path, selector)
        assert path.endswith(".pkl")

    def get(self):
        data = pickle.load(self.handle)
        keys = self.selector.keys or data.keys()
        return OrderedDict((x, data[x][self.selector.slicing]) for x in keys)


class Hdf5Reader(GenericReader):
    """Read .h5 file in hdf5 format"""

    def __init__(self, path, selector):
        """Store path and selector"""
        super().__init__(path, selector)
        assert path.endswith(".h5")

    def _open(self):
        self.handle = h5py.File(self.path, "r")

    def _get_column(self, key):
        """Get one column"""
        data = self.handle[key][self.selector.slicing]
        if data.dtype == np.object:
            return data.tolist()
        return data

    def get(self):
        if self.selector.keys is None:
            keys = list(self.handle.attrs['header'])
        else:
            keys = self.selector.keys
        return OrderedDict((x, self._get_column(x)) for x in keys)


class LstReader(GenericReader):
    """Read .lst file in list format"""

    def __init__(self, path, selector):
        """Store path and selector"""
        super().__init__(path, selector)
        assert path.endswith(".lst")

        hit = re.search(r'\[([a-zA-Z0-9_|]+)\]', path)
        if not hit:
            self.path = {"item": path}
        else:
            hit = hit.group(1)
            self.path = OrderedDict((x, path.replace("[%s]" % hit, x)) for x in hit.split("|"))

    def _open(self):
        self.handle = {x: open(y, 'rb') for x, y in self.path.items()}

    def close(self):
        for handle in self.handle.values():
            handle.close()

    def _get_column(self, key):
        """Get one column"""
        acc = []
        index = 0
        for line in self.handle[key]:
            if index % self.selector.partitions == self.selector.slice:
                acc.append(self.str2collection(line.decode("utf8").strip("\n")))

            index += 1
            if self.selector.limit and index >= self.selector.limit:
                break

        return acc

    def get(self):
        if self.selector.keys is None:
            keys = list(self.path.keys())
        else:
            keys = self.selector.keys
        return OrderedDict((x, self._get_column(x)) for x in keys)


READERS = {
    "tsv": TsvReader,
    "csv": TsvReader,
    "pkl": PklReader,
    "h5": Hdf5Reader,
    "lst": LstReader,
}

def initialize_file_reader(path, **kwargs):
    """Given path, initialize the correct reader based on the path suffix. All kwargs arguments
        are treated as selector opts."""
    base, suffix = path.rsplit(".", 1)
    if suffix in ["gz", "xz"]:
        suffix = base.rsplit(".", 1)[1]

    if suffix not in READERS:
        raise ValueError("Suffix '%s' is not supported ('%s')" % (suffix, path))
    return READERS[suffix](path, InputSelector(**kwargs))
