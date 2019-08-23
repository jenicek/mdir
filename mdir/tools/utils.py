import copy
import socket
import re
import io
import hashlib
import pickle
from urllib.request import urlopen


def get_dataset_params(params, net_defaults):
    params = copy.deepcopy({**net_defaults, **params})

    # Re-map data
    if "image_dir" in params["dataset"] \
            and (params["dataset"]["image_dir"] == "/mnt/fry/image/" or params["dataset"]['image_dir'] == "/mnt/fry/image/*") \
            and socket.gethostname() in {"castor", "pollux", "lcgpu"}:
        params["dataset"]["image_dir"] = "/data/data/radenfil/Research/deepsiamac/data/train/ims/" + \
                            ("*" if params["dataset"]['image_dir'][-1] == "*" else "")
        print("Using data from %s" % params["dataset"]["image_dir"])
    return params


def indent(string, indent=1):
    return string.replace("\n", "\n" + "    " * indent)


def validate(content, path):
    match = re.search(r'.*-([a-f0-9]{8,})\.[a-zA-Z0-9]{2,}$', path)
    if match:
        stored_hsh = match.group(1)
        computed_hsh = hashlib.sha256(content).hexdigest()[:len(stored_hsh)]
        if computed_hsh != stored_hsh:
            raise ValueError("Computed hash '%s' is not consistent with stored hash '%s'" \
                    % (computed_hsh, stored_hsh))

def load_url(url):
    with urlopen(url) as handle:
        loaded = io.BytesIO(handle.read())

    validate(loaded.getvalue(), url)
    return loaded


def load_path(path):
    assert path.endswith(".pkl"), "Cannot load anything else than pickle at the moment"
    if path.startswith("http://") or path.startswith("https://"):
        return pickle.load(load_url(path))

    with open(path, 'rb') as handle:
        return pickle.load(handle)
