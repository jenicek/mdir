import os
import sys
import time
import warnings
import abc
from pathlib import Path
import pickle
from PIL import Image, ImageOps
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import plots
from daan.presentation import presentation as pres

matplotlib.rcParams.update({'font.size': matplotlib.rcParams['font.size'] * 2})

# Valid dtypes:
#
# - scalar/loss (curve + hist)
# - scalar/score (curve + hist)
# - scalar/time (distribution)
# - weight/param (histogram only)
# - weight/grad (histogram only)
# - blob (image - rgb, rgba, gray; binary - response, vector)


#
# Abstract
#

class EventSink(abc.ABC):

    @abc.abstractmethod
    def load_epochs_data(self, epochs, consts):
        pass

    @abc.abstractmethod
    def register_epoch_data(self, epoch, data, consts):
        pass


class EventStreamer(abc.ABC):

    @abc.abstractmethod
    def add_row(self, epoch, timestamp, relative_iteration, epoch_size, key, data, dtype):
        pass


#
# Fundamental implementations
#

class MetadataKeeper(EventSink):

    aggregations = {
        "avg": "_avg.4",
        "sum": "_sum.1",
        None: "",
    }

    def __init__(self, dataroot):
        self.epochs = []
        self.data = {}
        self.keys = {}

    def load_epochs_data(self, epochs, consts):
        assert not self.data
        for i, data in enumerate(epochs):
            self.register_epoch_data(i, data, consts)
        return self

    def register_epoch_data(self, epoch, data, consts):
        assert epoch >= 0
        self.epochs.append(epoch)
        for key, item in data.items():
            if key in self.keys:
                assert self.keys[key] == item["data"].keys()
                continue

            self.keys[key] = item["data"].keys()
            if item["dtype"].startswith("scalar/"):
                for subkey, subitem in item["data"].items():
                    if not isinstance(subitem, (list, np.ndarray)):
                        aggr = None
                    else:
                        aggr = "avg" if item["dtype"] in {"scalar/loss", "scalar/score"} else "sum"

                    new_key = key + ":" + subkey + self.aggregations[aggr]
                    self.data[key, subkey] = {
                        "iteration_density": [],
                        "dtype": item["dtype"],
                        "aggr": aggr,
                        "key": new_key,
                        "epochs": [],
                        "data": [],
                    }

        for key, item in self.data.items():
            if key[0] not in data:
                continue

            value = np.array(data[key[0]]["data"][key[1]])
            iteration_density = None
            if item["aggr"] is not None:
                value = value[~np.isnan(value)]
                value = {"avg": np.mean, "sum": np.sum}[item["aggr"]](value)
                if data[key[0]]["relative_iteration"] is not None:
                    iteration_density = len(data[key[0]]["relative_iteration"]) / data[key[0]]["epoch_size"]

            item["iteration_density"].append(iteration_density)
            item["epochs"].append(epoch)
            item["data"].append(value)

    def metric(self, data_key, item_key):
        return self.data[data_key,item_key]["data"]

    def metadata(self):
        return {y["key"]: y["data"] for y in self.data.values() if y["dtype"] in {"scalar/loss", "scalar/score"}}

    def is_last_best(self, key):
        if isinstance(key, str):
            key = tuple(key.split(":"))
        assert isinstance(key, tuple), key

        if key == ("epoch",):
            return True
        elif key not in self.data or self.data[key]["epochs"][-1] != self.epochs[-1]:
            return False

        if self.data[key]["dtype"] == "scalar/score":
            return max(self.data[key]["data"]) == self.data[key]["data"][-1]
        return min(self.data[key]["data"]) == self.data[key]["data"][-1]

    def best_epoch(self, key):
        if isinstance(key, str):
            key = tuple(key.split(":"))
        assert isinstance(key, tuple)

        if key == ("epoch",):
            return {"index": self.epochs[-1], "metric_avg.3": self.epochs[-1], "key": "epoch"}
        elif key not in self.data:
            return None

        if self.data[key]["dtype"] == "scalar/score":
            index = np.argmax(self.data[key]["data"])
        else:
            index = np.argmin(self.data[key]["data"])
        return {"index": self.data[key]["epochs"][index], "metric_avg.3": self.data[key]["data"][index], "key": self.data[key]["key"]}

    def errors(self):
        errors = []

        if self.epochs != list(range(len(self.epochs))):
            errors.append({"message": "Non-standard epoch sequence used", "data": self.epochs})

        iteration_density = {"%s@epoch_%s" % (x["key"], z): y for x in self.data.values() for y, z in zip(x["iteration_density"], x["epochs"]) if y != 1}
        if iteration_density:
            errors.append({"message": "Some keys have incomplete iteration coverage", "data": iteration_density})

        epoch_coverage = {x["key"]: x["epochs"] for x in self.data.values() if x["epochs"] != self.epochs}
        if epoch_coverage:
            errors.append({"message": "Some keys have incomplete epoch coverage", "data": epoch_coverage})

        return errors


class EpochEventAccumulator(EventStreamer):

    folder_name = "blobs"
    histogram_bins = 200
    dtypes = {"scalar/loss", "scalar/score", "scalar/time", "weight/param", "weight/grad", "blob"}
    suffixes = {
        "image:rgb": "png",
        "image:rgba": "png",
        "image:gray": "png",
        "response": "tiff",
        "vector": "pkl",
    }

    def __init__(self, dataroot):
        self.datapath = (Path(dataroot) / self.folder_name) if dataroot is not None else None
        self.epoch = None
        self.accumulator = []
        self.datapath_created = False

    @staticmethod
    def _store_single_blob(img, path, suffix):
        if isinstance(img, torch.Tensor):
            img = img.detach().numpy()
            if len(img.shape) == 3 and img.shape[0] == 1:
                img = img[0]
            elif len(img.shape) == 3:
                img = img.transpose((1, 2, 0))
        with path.open("wb") as handle:
            if suffix == "png":
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img *= 255
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                assert len(img.shape) in {2, 3}, img.shape
                Image.fromarray(img).save(handle, format=suffix.upper())
            elif suffix == "tiff":
                assert len(img.shape) == 2, img.shape
                Image.fromarray(img).save(handle, format=suffix.upper())
            else:
                pickle.dump(img, handle)

        return str(path)

    def _store_blob(self, fname_piece, key, data):
        if self.datapath is None:
            # Ignore silently
            for subkey, value in data.items():
                del value["data"]
                value["path"] = None
            return data

        if not self.datapath_created:
            self.datapath.mkdir(parents=True, exist_ok=True)
            self.datapath_created = True
        for subkey, value in data.items():
            suffix = self.suffixes[value["dtype"]]
            fname = "%s:%s:%s.%s" % (key.replace("/", "_"), subkey.replace("/", "_"), fname_piece, suffix)
            value["path"] = self._store_single_blob(value.pop("data"), self.datapath / fname, suffix)
        return data

    def _generate_hist(self, data):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                value = value.cpu().detach().numpy().copy()
            hist = np.histogram(value, bins=self.histogram_bins, density=False)
            data[key] = ((hist[1][:-1] + hist[1][1:]) / 2, hist[0])
        return data

    def add_row(self, epoch, timestamp, relative_iteration, epoch_size, key, data, dtype):
        assert epoch >= 0
        assert isinstance(data, dict), type(data)
        assert dtype in self.dtypes, dtype

        if dtype == "blob":
            rel_iter = "null" if relative_iteration is None else relative_iteration
            data = self._store_blob("%s:%s" % (epoch, rel_iter), key, data)
        elif dtype.startswith("weight/"):
            data = self._generate_hist(data)

        if self.epoch is None:
            self.epoch = epoch
        elif self.epoch != "error" and self.epoch != epoch:
            warnings.warn("inconsistent epoch (%s != %s)" % (epoch, self.epoch))
            self.epoch = "error"

        row = {"timestamp": timestamp, "relative_iteration": relative_iteration,
               "epoch_size": epoch_size, "key": key, "data": data, "dtype": dtype}
        self.accumulator.append(row)

    def aggregate(self):
        aggregated = {}
        for row in self.accumulator:
            if row["relative_iteration"] is None:
                assert row["key"] not in aggregated
                aggregated[row["key"]] = row
            elif row["key"] not in aggregated:
                data = {x: [y] for x, y in row["data"].items()}
                aggregated[row["key"]] = {"dtype": row["dtype"], "epoch_size": row["epoch_size"],
                                          "data": data, "relative_iteration": [row["relative_iteration"]],
                                          "timestamp": [row["timestamp"]]}
            else:
                assert aggregated[row["key"]]["dtype"] == row["dtype"], \
                        "%s: %s != %s" % (row["key"], aggregated[row["key"]]["dtype"], row["dtype"])
                assert aggregated[row["key"]]["epoch_size"] == row["epoch_size"], \
                        "%s: %s != %s" % (row["key"], aggregated[row["key"]]["epoch_size"], row["epoch_size"])
                assert aggregated[row["key"]]["data"].keys() == row["data"].keys()

                for key, value in row["data"].items():
                    aggregated[row["key"]]["data"][key].append(value)
                aggregated[row["key"]]["relative_iteration"].append(row["relative_iteration"])
                aggregated[row["key"]]["timestamp"].append(row["timestamp"])

        # Postprocessing
        for key, item in aggregated.items():
            if item["relative_iteration"] is None:
                continue

            if item["dtype"].startswith("scalar/"):
                for subkey, subitem in item["data"].items():
                    item["data"][subkey] = np.array(subitem)
            item["relative_iteration"] = np.array(item["relative_iteration"])
            item["timestamp"] = np.array(item["timestamp"])

        return aggregated


class ConstEventAccumulator(EpochEventAccumulator):

    def __init__(self, dataroot, consts):
        super().__init__(dataroot=dataroot)
        self.consts = consts

    def add_const(self, key, data, dtype):
        assert dtype in self.dtypes, dtype

        if dtype == "blob":
            # Do not repeatadly store the blob
            if key in self.consts:
                assert self.consts[key]["dtype"] == dtype, "%s: %s != %s" % (key, self.consts[key]["dtype"], dtype)
                return
            data = self._store_blob("const", key, data)
        elif dtype.startswith("weight/"):
            data = self._generate_hist(data)

        const = {"dtype": dtype, "data": data}
        if key in self.consts:
            assert self.consts[key] == const, "%s: %s != %s" % (key, self.consts[key], const)
            return

        self.consts[key] = const


#
# Regular implementations
#

class DebugPrinter(EventStreamer):

    def __init__(self, print_each=1, print_each_val=None, key_suffix="learning/loss:total", dataroot=None):
        self.print_each = print_each
        self.print_each_val = print_each_val if print_each_val is not None else print_each
        self.key_suffix = tuple(key_suffix.split(":"))
        assert len(self.key_suffix) == 2
        self.acc = {}
        self.iteration_timestamps = {}

    def add_row(self, epoch, timestamp, relative_iteration, epoch_size, key, data, dtype):
        if not self.print_each:
            return

        if key.endswith(self.key_suffix[0]) and self.key_suffix[1] in data:
            stage = key.split("/", 1)[0].capitalize()
            loss = data[self.key_suffix[1]]
            timestamp = time.time()
            relative_iteration1 = relative_iteration + 1 if relative_iteration is not None else None

            if (epoch, stage) not in self.acc:
                self.acc[(epoch, stage)] = {"first_timestamp": timestamp, "last_timestamp": None, "n": 0, "loss_sum": 0}

            history = self.acc[(epoch, stage)]
            history["last_timestamp"] = timestamp
            history["n"] += 1
            history["loss_sum"] += loss

            print_each = self.print_each_val if stage.startswith("Val") else self.print_each
            if relative_iteration1 is None or relative_iteration1 % print_each == 0 \
                    or relative_iteration1 == epoch_size:

                additional_timing = ""
                self.iteration_timestamps[(epoch, stage, relative_iteration1)] = timestamp
                previous_key = (epoch-1, stage, relative_iteration1)
                if previous_key in self.iteration_timestamps:
                    additional_timing += ", %d m/e" % round((timestamp - self.iteration_timestamps[previous_key]) / 60)

                sys.stderr.write("%s [%0.2d][%0.3d/%0.3d]: %.2f (%.2f), (%.2f s/b%s)\n" % \
                        (stage, epoch+1, relative_iteration1, epoch_size, loss, history["loss_sum"] / history["n"],
                        (timestamp - history["first_timestamp"]) / (history["n"] - 1), additional_timing))


class Tensorboard(EventStreamer, EventSink):

    folder_name = "tensorboard"

    def __init__(self, dataroot):
        from tensorboardX import SummaryWriter

        self.writer = SummaryWriter(os.path.join(dataroot, self.folder_name))
        self.absolute_iteration_counters = {}

    def _add_row(self, key, data, dtype, iteration):
        if dtype.startswith("scalar/"):
            for subkey, value in data.items():
                self.writer.add_scalar("%s/%s" % (key, subkey), value, iteration)
        elif dtype.startswith("weight/"):
            for subkey, value in data.items():
                self.writer.add_histogram("%s/%s" % (key, subkey), value, iteration, 'auto')
        elif dtype == "blob":
            for subkey, value in data.items():
                if value["dtype"] == "gray":
                    self.writer.add_image("%s/%s" % (key, subkey), np.repeat(np.expand_dims(value["data"], axis=2), 3, axis=2), iteration)
                if value["dtype"] == "rgb" or value["dtype"] == "rgba":
                    self.writer.add_image("%s/%s" % (key, subkey), value["data"], iteration)

    def add_row(self, epoch, timestamp, relative_iteration, epoch_size, key, data, dtype):
        if epoch is None:
            iteration = 0
        elif relative_iteration is None:
            iteration = epoch
        else:
            iteration = self.absolute_iteration_counters.get(key, 0) + relative_iteration
        self._add_row(key, data, dtype, iteration)

    def register_epoch_data(self, epoch, data, consts):
        for key, item in data.items():
            if item["relative_iteration"] is None:
                continue

            if key not in self.absolute_iteration_counters:
                self.absolute_iteration_counters[key] = 0
            self.absolute_iteration_counters[key] += item["epoch_size"]
            if item["dtype"] in {"scalar/loss", "scalar/score"}:
                for subkey, value in item["data"].items():
                    self.writer.add_scalar("%s/%s_avg" % (key, subkey), np.mean(value), epoch)
            elif item["dtype"] == "scalar/time":
                for subkey, value in item["data"].items():
                    self.writer.add_scalar("%s/%s_sum" % (key, subkey), np.sum(value), epoch)

    def load_epochs_data(self, epochs, consts):
        for epoch in epochs:
            for key, item in epoch.items():
                if item["relative_iteration"] is None:
                    continue

                if key not in self.absolute_iteration_counters:
                    self.absolute_iteration_counters[key] = 0
                self.absolute_iteration_counters[key] += item["epoch_size"]
        return self


class HtmlReport(EventSink):

    folder_name = "htmlreport"

    def __init__(self, dataroot):
        self.dataroot = dataroot
        self.data = {}
        os.makedirs(os.path.join(dataroot, self.folder_name), exist_ok=True)

    def load_epochs_data(self, epochs, consts):
        assert not self.data
        for i, data in enumerate(epochs):
            self._store_epoch_data(i, data, consts)
        self.render(len(epochs)-1)
        return self

    def _store_epoch_data(self, epoch, data, consts):
        assert epoch >= 0
        for key, item in data.items():
            *key, key2 = key.split("/", 2)
            key = "/".join(key)
            if key not in self.data:
                self.data[key] = {}

            for subkey, subitem in item["data"].items():
                subkey = "%s/%s" % (key2, subkey)
                if subkey not in self.data[key]:
                    self.data[key][subkey] = {"data": [], "subtype": item["dtype"].rsplit("/", 1)[1] if "/" in item["dtype"] else ""}
                if item["dtype"].startswith("scalar/"):
                    if not isinstance(subitem, (list, np.ndarray)):
                        self.data[key][subkey]["plot_type"] = "curve"
                        self.data[key][subkey]["data"].append((epoch+1, subitem))
                    else:
                        if not isinstance(subitem, np.ndarray):
                            subitem = np.array(subitem)
                        subitem = subitem[~np.isnan(subitem)]
                        values, bins = np.histogram(subitem, bins=20)
                        centers = (bins[1:] + bins[:-1])/2
                        self.data[key][subkey]["plot_type"] = "distribution"
                        self.data[key][subkey]["data"].append((epoch+1, centers, values, np.mean(subitem)))
                elif item["dtype"].startswith("weight/"):
                    self.data[key][subkey]["plot_type"] = "histogram"
                    for i, subitem_item in enumerate(subitem):
                        self.data[key][subkey]["data"].append((epoch + (item["relative_iteration"][i]+1)/item["epoch_size"],) + subitem_item)
                elif item["dtype"] == "blob":
                    self.data[key][subkey]["plot_type"] = "thumbnail_set"
                    for i, subitem_item in enumerate(subitem):
                        self.data[key][subkey]["data"].append({**subitem_item, "epoch": epoch, "iteration": item["relative_iteration"][i],
                                                               "thumbnail": self._square_thumbnail(subitem_item["path"], 200)})

        for key, item in consts.items():
            *key, key2 = key.split("/", 2)
            key = "/".join(key)
            if key not in self.data and item["dtype"] == "blob":
                self.data[key] = {}
                for subkey, subitem in item["data"].items():
                    subkey = "%s/%s" % (key2, subkey)
                    self.data[key][subkey] = {**subitem, "plot_type": "thumbnail",
                                              "thumbnail": self._square_thumbnail(subitem["path"], 200)}

    def register_epoch_data(self, epoch, data, consts):
        self._store_epoch_data(epoch, data, consts)
        self.render(epoch)

    def _square_thumbnail(self, path, size):
        thumb = "%s.thumb.%s" % tuple(os.path.basename(path).rsplit(".", 1))
        if not os.path.exists(os.path.join(self.dataroot, self.folder_name, thumb)):
            img = Image.open(path)
            # Pad
            if min(img.size) < size:
                diff0, diff1 = max(size - img.size[0], 0), max(size - img.size[1], 0)
                newimg = Image.new('RGBA', (size, size), (255, 255, 255, 0))
                newimg.paste(img, (diff0 // 2, diff1 // 2))
                img = newimg
            # Crop
            diff0, diff1 = (img.size[0] - min(img.size))/2, (img.size[1] - min(img.size))/2
            img = img.crop((diff0, diff1, min(img.size) + diff0, min(img.size) + diff1))
            img.thumbnail((size, size))
            img.save(os.path.join(self.dataroot, self.folder_name, thumb))
        return thumb

    def render(self, epoch):
        name = os.path.basename(os.path.dirname(os.path.abspath(self.dataroot)))
        html = {"name": "<div style='word-break: break-word;'>Epoch %s of %s</div>" \
                % (epoch+1, name), "data": [], "type": "rows"}
        order = {"train/learning": 0, "val/learning": 1, "train/net": 2, "net": 3, "train/data": 4}
        sets = {}
        for key, item in sorted(self.data.items(), key=lambda x: order.get(x[0], 100)):
            section = []
            for subkey, subitem in item.items():
                fname = "%s_%s_%%s.png" % (key.replace("/", "_"), subkey.replace("/", "_"))
                if subitem["plot_type"] == "curve":
                    fname %= "plot"
                    self.store_plot(fname, subitem["data"], subitem["subtype"])
                elif subitem["plot_type"] == "distribution":
                    fname %= "dist"
                    self.store_distribution(fname, subitem["data"], subitem["subtype"])
                elif subitem["plot_type"] == "histogram":
                    fname %= "hist"
                    self.store_histogram(fname, subitem["data"], subitem["subtype"])

                elif subitem["plot_type"] == "thumbnail":
                    fname = os.path.relpath(subitem["path"], os.path.join(self.dataroot, self.folder_name))
                    section.append({"type": "blocks", "name": subkey.replace("/", "<br />"),
                                    "data": [{"type": "image", "source": subitem["thumbnail"], "link": fname, "size": 200}]})
                    continue
                elif subitem["plot_type"] == "thumbnail_set":
                    if key not in sets:
                        sets[key] = {}
                    for singleimg in subitem["data"]:
                        name = "Epoch %s (iter %s)" % (singleimg["epoch"]+1, singleimg["iteration"]+1)
                        if name not in sets[key]:
                            sets[key][name] = []
                            section.append({"type": "blocks", "name": name, "data": sets[key][name]})
                        fname = os.path.relpath(singleimg["path"], os.path.join(self.dataroot, self.folder_name))
                        sets[key][name].append({"type": "blocks", "name": subkey.replace("/", "<br />"),
                                                "data": [{"type": "image", "source": singleimg["thumbnail"], "link": fname, "size": 200}]})
                    continue
                else:
                    continue

                h_name = "<div style='max-width: 300px; word-break: break-word;'>%s</div>" % subkey.replace("/", "<br />")
                section.append({"type": "blocks", "name":  h_name,
                                "data": [{"type": "image", "source": fname, "link": fname, "size": 300}]})
            html["data"].append({"name": key, "data": section, "type": "blocks", "css": "margin: 0 3pt 0 3pt;"})

        with open(os.path.join(self.dataroot, self.folder_name, "index.html"), "w") as handle:
            css = """
            @media only screen and (min-resolution: 200dpi) {
                body { zoom: 0.65; }
            }
            """
            handle.write(pres.Document().struct2html(html, css=css))

    def store_plot(self, fname, data, ylabel):
        plt.figure(figsize=(6, 4))
        # plt.xlabel("epoch")
        plt.ylabel(ylabel)

        plots.plot_curve(data, plt.gca())

        plt.savefig(os.path.join(self.dataroot, self.folder_name, fname), transparent=True,
                    bbox_inches='tight')
        plt.close()

    def store_distribution(self, fname, data, ylabel):
        plt.figure(figsize=(6, 4))
        # plt.xlabel("epoch")
        plt.ylabel(ylabel)
        ax = plt.gca()

        if len(data[0]) == 4:
            plots.plot_curve([(i, z) for i, x, y, z in data], plt.gca())
            plots.plot_distribution([x[:3] for x in data], plt.gca())
        else:
            plots.plot_distribution(data, plt.gca())

        plt.savefig(os.path.join(self.dataroot, self.folder_name, fname), transparent=True,
                    bbox_inches='tight')
        plt.close()

    def store_histogram(self, fname, data, ylabel):
        plt.figure(figsize=(6, 5))
        plt.xlabel(ylabel)
        # plt.ylabel("epoch")
        ax = plt.gca()

        plots.plot_histogram(data, plt.gca())

        plt.savefig(os.path.join(self.dataroot, self.folder_name, fname), transparent=True,
                    bbox_inches='tight')
        plt.close()


EVENTPROCESSORS = {
    "progress": DebugPrinter,
    "tensorboard": Tensorboard,
    "htmlreport": HtmlReport,
}


#
# Wrapping code
#

class EventBroker:

    def __init__(self, processors, dataroot, consts, data):
        self.params = {"processors": processors, "dataroot": dataroot}
        self.data = data

        self.epoch_accumulator = EpochEventAccumulator(dataroot=dataroot)
        self.const_accumulator = ConstEventAccumulator(dataroot=dataroot, consts=consts)
        self.metadata = MetadataKeeper(dataroot=dataroot).load_epochs_data(data, consts)
        self.streamers = []
        self.sinks = []
        for processor in processors:
            proc_cls = EVENTPROCESSORS[processor]
            if isinstance(processors[processor], dict):
                proc = proc_cls(**processors[processor], dataroot=dataroot)
            else:
                proc = proc_cls(processors[processor], dataroot=dataroot)

            if isinstance(proc, EventStreamer):
                self.streamers.append(proc)
            if isinstance(proc, EventSink):
                self.sinks.append(proc.load_epochs_data(data, consts))
            if not isinstance(proc, (EventSink, EventStreamer)):
                raise ValueError("Unsupported processor type '%s'" % type(proc))

    @classmethod
    def initialize(cls, processors, dataroot):
        return cls(processors, dataroot, {}, [])

    def register_data(self, epoch, relative_iteration, epoch_size, key, data, dtype):
        params = {"epoch": epoch, "timestamp": time.time(), "relative_iteration": relative_iteration,
                  "epoch_size": epoch_size, "key": key, "data": data, "dtype": dtype}

        for streamer in self.streamers:
            streamer.add_row(**params)
        # After streamers because changes the data
        if epoch is None:
            self.const_accumulator.add_const(key=key, data=data, dtype=dtype)
        else:
            self.epoch_accumulator.add_row(**params)

    def close_epoch(self):
        epoch = self.epoch_accumulator.epoch
        assert len(self.data) == epoch, "%s != %s" % (len(self.data), epoch)
        epoch_data = self.epoch_accumulator.aggregate()

        self.metadata.register_epoch_data(epoch, epoch_data, self.const_accumulator.consts)
        for sink in self.sinks:
            sink.register_epoch_data(epoch, epoch_data, self.const_accumulator.consts)

        self.data.append(epoch_data)
        self.epoch_accumulator = EpochEventAccumulator(dataroot=self.params["dataroot"])

    #
    # Load and save
    #

    def state_dict(self):
        return {
            "name": self.__class__.__name__,
            "params": self.params,
            "consts": self.const_accumulator.consts,
            "data": self.data,
        }

    @classmethod
    def initialize_from_state(cls, state_dict, params):
        assert state_dict["name"] == cls.__name__
        if params is not None:
            assert params['processors'] == state_dict["params"]['processors'], "%s != %s" % (str(params['processors']), str(state_dict["params"]['processors']))
            state_dict['params']['dataroot'] = params['dataroot']

        return cls(**state_dict["params"], consts=state_dict["consts"], data=state_dict["data"])


EVENTBROKERS = {
    "EventBroker": EventBroker,
}

def initialize_processor(params, dataroot, state=None):
    proc = EVENTBROKERS[params.pop("type", "EventBroker")]
    if state is None:
        return proc.initialize(processors=params, dataroot=dataroot)
    return proc.initialize_from_state(state, {"processors": params, "dataroot": dataroot})
