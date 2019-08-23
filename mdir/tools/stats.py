import time
import os
import sys
import psutil
from pathlib import Path
import torch.cuda

from daan.data import sysstats


class AverageMeter:

    def __init__(self, stage, loader_size, debug):
        self.stage = stage
        self.loader_size = loader_size
        self.debug = (100 if debug else False) if isinstance(debug, bool) else debug
        self.time0 = time.time()
        self.sum = 0
        self.count = 0

    def update(self, iteration, loss):
        if loss:
            self.sum += loss
        self.count += 1
        iteration1 = iteration + 1
        places = len(str(self.loader_size))
        if self.debug and iteration == 0:
            sys.stderr.write("\r%s: [%0{0}d/%0{0}d] ".format(places) % (self.stage, iteration1, self.loader_size))
        elif self.debug and (iteration1 % self.debug == 0 or iteration1 == self.loader_size):
            avg_time = (time.time() - self.time0) / iteration1
            if loss:
                sys.stderr.write("\r%s: [%0{0}d/%0{0}d]: %.3f (elapsed %02dm/%02dm)  ".format(places) % (self.stage, iteration1, self.loader_size, self.sum / self.count, avg_time*iteration1/60, avg_time*self.loader_size/60))
            else:
                sys.stderr.write("\r%s: [%0{0}d/%0{0}d] elapsed %02dm/%02dm  ".format(places) % (self.stage, iteration1, self.loader_size, avg_time*iteration1/60, avg_time*self.loader_size/60))
            if iteration1 == self.loader_size:
                sys.stderr.write("\n")
        return self

    def total_stats(self):
        total_time = time.time() - self.time0
        stats = {"total_time": int(total_time), "avg_time": total_time/self.loader_size}
        if self.sum:
            stats["avg_loss"] = self.sum / self.count
        return stats


class StopWatch:

    def __init__(self):
        self.timings = {}
        self.time0 = time.time()
        self.time_reset = self.time0

    def reset(self, include_total=True):
        timings = self.timings
        self.timings = {}
        self.time0 = time.time()

        if include_total:
            timings["total_s"] = self.time0 - self.time_reset
        self.time_reset = self.time0
        return timings

    def lap(self, name):
        curtime = time.time()
        self.timings[name] = curtime - self.time0
        self.time0 = curtime


class ResourceUsage:
    """Keep track of resource usage by current process"""

    def __init__(self, accumulated=None):
        """Initialize empty resources dict"""
        self.accumulated = accumulated
        self.resources = {}

    @staticmethod
    def initialize():
        return ResourceUsage(None)

    def take_current_stats(self):
        """Store non-cumulative stats (e.g. current memory usage). Currently, RAM and GPU memory
            usage is stored."""
        proc = psutil.Process()
        # RAM memory
        self.resources['ram_memory_gib'] = round(proc.memory_info().vms / 2**30, 3)
        # GPU memory
        if torch.cuda.is_available():
            gpu_mem = sysstats.NvidiaStats.memory_usage_by_pid().get(os.getpid(), None)
            self.resources['gpu'] = {
                'memory_nvidia_gib': round(gpu_mem / 2**10, 3) if gpu_mem else gpu_mem,
                'memory_torch_gib': round(torch.cuda.memory_allocated() / 2**30, 3),
            }
        # Ignored: num_fds, num_handles, threads
        return self

    def get_cumulative_stats(self):
        """Return cumulative stats (e.g. cpu time). Currently, CPU and IO stats are stored."""
        proc = psutil.Process()
        stats = {}
        with proc.oneshot(): # Retrieve all process info at once
            # CPU
            accum_cpu = self.accumulated["cpu"] if self.accumulated is not None else {}
            cpu = proc.cpu_times()
            stats['cpu'] = {
                'user_s': int(cpu.user) + accum_cpu.get("user_s", 0),
                'system_s': int(cpu.system) + accum_cpu.get("system_s", 0),
                'children_user_s': int(cpu.children_user) + accum_cpu.get("children_user_s", 0),
                'children_system_s': int(cpu.children_system) + accum_cpu.get("children_system_s", 0),
                'proc_wall_s': int(time.time() - proc.create_time()) + accum_cpu.get("proc_wall_s", 0),
            }
            stats['cpu']['tree_used_s'] = stats['cpu']['user_s'] + stats['cpu']['system_s'] + \
                    stats['cpu']['children_user_s'] + stats['cpu']['children_system_s']
            stats['cpu']['avg_cores'] = round(stats['cpu']['tree_used_s'] / stats['cpu']['proc_wall_s'], 1)

            # IO
            accum_io = self.accumulated["io"] if self.accumulated is not None else {}
            io_count = proc.io_counters()
            stats['io'] = {
                'read_count': io_count.read_count + accum_io.get("read_count", 0),
                'write_count': io_count.write_count + accum_io.get("write_count", 0),
                'read_gib': round(io_count.read_bytes / 2**30 + accum_io.get("read_gib", 0), 3),
                'write_gib': round(io_count.write_bytes / 2**30 + accum_io.get("write_gib", 0), 3),
            }
        # Ignored: num_ctx_switches
        return stats

    def get_resources(self):
        return {**self.resources, **self.get_cumulative_stats()}

    def state_dict(self):
        return {
            "name": self.__class__.__name__,
            "params": {},
            "cumulative_stats": self.get_cumulative_stats(),
        }

    @staticmethod
    def initialize_from_state(state):
        assert state["name"] == ResourceUsage.__name__
        assert not state["params"]
        return ResourceUsage(state["cumulative_stats"])


class CodeVersion:

    def __init__(self):
        self.versions = {
            "mdir_git": self.git_head_state("mdir"),
        }

    @staticmethod
    def git_head_state(module_name):
        if not hasattr(sys.modules.get(module_name, None), "__file__"):
            return None

        try:
            git_path = Path(sys.modules[module_name].__file__).parent.parent / ".git"
            with (git_path / "HEAD").open() as handle:
                head_content = handle.read().strip()

            if head_content.startswith("ref:"):
                head_ref = head_content[len("ref:"):].strip()
                with (git_path / head_ref).open() as handle:
                    commit = handle.read().strip()
                return {"commit": commit, "head_ref": head_ref}

            return {"commit": head_content, "head_ref": None}

        except FileNotFoundError:
            return None
