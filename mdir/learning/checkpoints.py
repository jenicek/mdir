import os
from pathlib import Path
import torch

from ..tools.utils import load_url

SUFFIX_NOTRAIN = "_notrain.pth"
SUFFIX_FROZEN = "_frozen.pth"
SUFFIX_EPOCH = "_epoch_%02d.pth"
SUFFIX_BEST_SO_FAR = "_bestsofar.pth"
SUFFIX_BEST = "_best.pth"
SUFFIX_LAST = "_last.pth"

FNAME_TRAINING = "learning_epoch_%02d.pth"


class Checkpoints:

    def __init__(self, directory, store_every, checkpoint_every):
        self.directory = Path(directory) / "epochs"
        self.store_every = store_every
        self.checkpoint_every = checkpoint_every

    def save_notrain(self, networks_state):
        for key, state in networks_state.items():
            assert "/" not in key
            notrain_path = self.directory / (key + SUFFIX_NOTRAIN)
            torch.save(state, notrain_path)
            (self.directory / (key + SUFFIX_BEST)).symlink_to(key + SUFFIX_NOTRAIN)
            (self.directory / (key + SUFFIX_LAST)).symlink_to(key + SUFFIX_NOTRAIN)

    def save_epoch(self, networks_state, training_state, epoch, is_best, is_last):
        assert epoch >= 0
        epoch1 = epoch + 1
        is_checkpointed = (self.checkpoint_every > 0 and epoch1 % self.checkpoint_every == 0) or is_last
        is_stored = self.store_every > 0 and epoch1 % self.store_every == 0
        if is_checkpointed:
            last_checkpoint = epoch - (epoch1 % self.checkpoint_every or self.checkpoint_every)
            last_is_stored = self.store_every > 0 and (last_checkpoint + 1) % self.store_every == 0

        if not self.directory.exists():
            os.makedirs(self.directory)

        # Handle helper variables
        if len(networks_state) > 1:
            networks_state["net"]["_network_names"] = [x for x in networks_state if x != "net"]

        # Save networks
        for key, state in networks_state.items():
            assert "/" not in key
            if state["frozen"]:
                # Does not change anymore, symlink only
                frozen_path = self.directory / (key + SUFFIX_FROZEN)
                if not frozen_path.exists():
                    torch.save(state, frozen_path)

            # Save epoch
            epoch_path = self.directory / (key + SUFFIX_EPOCH % epoch1)
            if is_checkpointed or is_stored:
                if state["frozen"]:
                    epoch_path.symlink_to(key + SUFFIX_FROZEN)
                else:
                    torch.save(state, epoch_path)

            # Symlink / save best & last
            shortcut_paths = []
            if is_best:
                shortcut_paths.append(self.directory / (key + SUFFIX_BEST_SO_FAR))
            if is_last:
                shortcut_paths.append(self.directory / (key + SUFFIX_LAST))
            for spath in shortcut_paths:
                if spath.exists():
                    spath.unlink()
                if state["frozen"]:
                    spath.symlink_to(key + SUFFIX_FROZEN)
                elif is_checkpointed or is_stored:
                    spath.symlink_to(key + SUFFIX_EPOCH % epoch1)
                else:
                    torch.save(state, spath)

        # Save training
        if is_checkpointed or is_stored:
            training_path = self.directory / (FNAME_TRAINING % epoch1)
            training_path_tmp = self.directory / ((FNAME_TRAINING % epoch1) + ".tmp")
            torch.save(training_state, training_path_tmp)
            training_path_tmp.rename(training_path)
            if is_checkpointed and self.checkpoint_every and epoch >= self.checkpoint_every:
                (self.directory / (FNAME_TRAINING % (last_checkpoint+1))).unlink()

        # Remove unneeded networks
        for key, state in networks_state.items():
            best_path = self.directory / (key + SUFFIX_BEST_SO_FAR)
            if not best_path.exists():
                final_best = self.directory / (key + SUFFIX_BEST)
                if final_best.exists():
                    final_best.rename(best_path)
            # Remove previous if necessary
            if is_checkpointed and last_checkpoint >= 0 and not last_is_stored:
                previous_path = self.directory / (key + SUFFIX_EPOCH % (last_checkpoint+1))
                if previous_path == best_path.resolve():
                    best_path.unlink()
                    previous_path.rename(best_path)
                else:
                    previous_path.unlink()

            # Handle last
            if is_last:
                best_path.rename(self.directory / (key + SUFFIX_BEST))

    @staticmethod
    def _load_epoch_network(directory, suffix):
        network_state = {
            "net": torch.load(directory / ("net" + suffix), map_location=lambda storage, location: storage)
        }

        # Embedded networks
        assert "net" not in network_state["net"].get("_networks_included", {})
        network_state.update(network_state["net"].pop("_networks_included", {}))

        # External networks
        for name in network_state["net"].pop("_network_names", []):
            assert name not in network_state
            epoch_path = directory / (name + suffix)
            network_state[name] = torch.load(epoch_path, map_location=lambda storage, location: storage)
        return network_state

    def _load_epoch_training(self, suffix):
        return torch.load(self.directory / suffix)

    def load_latest_epoch(self, nepochs):
        if not self.directory.exists():
            return None

        for epoch in reversed(range(nepochs)):
            epoch1 = epoch + 1
            training_path = self.directory / (FNAME_TRAINING % epoch1)
            if training_path.exists():
                network = self._load_epoch_network(self.directory, SUFFIX_EPOCH % epoch1)
                training = self._load_epoch_training(FNAME_TRAINING % epoch1)
                return network, training

        return None

    @classmethod
    def load_network(cls, directory):
        if directory.startswith("http://") or directory.startswith("https://"):
            directory = load_url(directory)
        else:
            directory = Path(directory)
            if directory.is_dir():
                return cls._load_epoch_network(directory, SUFFIX_BEST)

        checkpoint = torch.load(directory, map_location=lambda storage, location: storage)
        assert "net" not in checkpoint.get("_networks_included", {})
        return {"net": checkpoint, **checkpoint.pop("_networks_included", {})}
