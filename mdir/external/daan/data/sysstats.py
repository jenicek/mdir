import os
import time
import subprocess

CMD_EXECUTOR = subprocess


class NvidiaStats: # pylint: disable=too-few-public-methods
    """Stats related to nvidia gpu. Keeps general properties of gpus and enables refreshing
        of gpu metrics and metrics of processes running on those gpus."""

    namespace = {"gpu": {"resource": "gpus", "process": "gpu_processes"}}

    def __init__(self):
        """Initalize empty process properties cache"""
        self.processes = {}

    @classmethod
    def memory_usage_by_pid(cls):
        """Return a dictionary where key is a process pid and value is its gpu memory usage in MB.
            Both are ints. Nvidia-smi is used to get the data."""
        _, table = cls._nvidia_smi("--query-compute-apps=pid,used_gpu_memory")
        return {int(x['pid']): int(x['used_gpu_memory']) for x in table}

    def get_current(self):
        """Refresh all gpus metrics and metrics of processes running on those gpus. Return two data
            namespaces in a dictionary: gpu and gpu_process. Each namespace contains a list
            of either gpus or processes running on those gpus. Each list element contains
            dictionary with two keys: 'values' and 'tags' with dictionaries containing data.
            Tags section can be used as primary key."""
        return {"gpu": self._get_gpu_metrics(), "gpu_process": self._get_gpu_processes()}

    @staticmethod
    def _nvidia_smi(query):
        """Execute given query using nvidia-smi command and return tuple with raw header and a table
            formatted as a list of dicts indexed by the header (units ommited)"""
        child = CMD_EXECUTOR.run(["nvidia-smi", query, "--format=csv,nounits"],
                                 stdout=CMD_EXECUTOR.PIPE)
        rows = [x.strip().split(", ") for x in child.stdout.decode("utf-8").strip().split("\n")]
        header = [x.split(" ")[0] for x in rows[0]]
        return rows[0], [dict(zip(header, x)) for x in rows[1:]]

    def _get_gpu_metrics(self):
        """Return gpu metrics in the format used by get_current function."""
        header, table = self._nvidia_smi("--query-gpu=pci.bus_id,index,utilization.gpu," \
                                         "utilization.memory,memory.free,memory.total")
        assert header == ['pci.bus_id', 'index', 'utilization.gpu [%]', 'utilization.memory [%]',
                          'memory.free [MiB]', 'memory.total [MiB]']

        rows = []
        for row in table:
            values = {x: int(row[x]) for x in ["memory.free", "memory.total", "utilization.gpu",
                                               "utilization.memory"]}
            tags = {x: int(row[x]) if x == "index" else row[x] for x in ["pci.bus_id", "index"]}
            rows.append({"values": values, "tags": tags})
        return rows

    def _get_gpu_processes(self):
        """Return pid and metrics of all processes running on each gpu present on the system
            in the format used by get_current function. Process information is cached, so only
            information about new processes is queried."""
        header, table = self._nvidia_smi("--query-compute-apps=gpu_bus_id,pid,used_gpu_memory")
        assert header == ['gpu_bus_id', 'pid', 'used_gpu_memory [MiB]']

        processes = {}
        rows = []
        for row in table:
            values = {x: int(row[x]) for x in ["used_gpu_memory"]}
            tags = {x: int(row[x]) if x == "pid" else row[x] for x in ["gpu_bus_id", "pid"]}
            process = self.processes[row["pid"]] if row["pid"] in self.processes \
                        else Process(row["pid"])
            values.update({"started": process.started})
            tags.update({"user": process.user, "cmd": process.cmd})
            processes[row["pid"]] = process
            rows.append({"values": values, "tags": tags})
        self.processes = processes
        return rows

    @staticmethod
    def format_view(view): # pylint: disable=too-many-locals
        """Return human-plausible strings for each gpu which can be directly printed"""
        ljust = lambda x, i: str(x).rjust(i-1).ljust(i)
        rjust = lambda x, i: str(x).ljust(i-1).rjust(i)
        bufs = [""]
        for gpu in view.aggr("index").sort("index").get("pci.bus_id"):
            # General info
            gpu_data = view.select({"pci.bus_id": gpu}).sort("timestamp")
            latest = gpu_data.get("timestamp")[-1]
            memory_free = gpu_data.get("memory.free")
            str_params = {"index": gpu_data.get("index")[-1],
                          "free_mem_avg": int(sum(memory_free) / len(memory_free)),
                          "free_mem_min": ljust(min(memory_free), 5),
                          "free_mem_max": rjust(max(memory_free), 5),
                          "total_mem": gpu_data.get("memory.total")[-1]}
            template = "## gpu %(index)s: %(free_mem_avg)+5s / %(total_mem)+5s " \
                    "(%(free_mem_min)s-%(free_mem_max)s)\n"
            bufs[-1] += template % str_params

            # Utilization info
            util_gpu = gpu_data.get("utilization.gpu")
            util_mem = gpu_data.get("utilization.memory")
            str_params = {"util_gpu_avg": int(sum(util_gpu) / len(util_gpu)),
                          "util_gpu_min": ljust(min(util_gpu), 3),
                          "util_gpu_max": rjust(max(util_gpu), 3),
                          "util_mem_avg": int(sum(util_mem) / len(util_mem)),
                          "util_mem_min": ljust(min(util_mem), 3),
                          "util_mem_max": rjust(max(util_mem), 3),
                          "filler": " " * 20}
            template = "\n%(filler)sgpu %(util_gpu_avg)+3s (%(util_gpu_min)s-%(util_gpu_max)s)" \
                    "\n%(filler)smem %(util_mem_avg)+3s (%(util_mem_min)s-%(util_mem_max)s)\n\n"
            bufs[-1] += template % str_params

            # Process info
            procs = view.select({"gpu_bus_id": gpu}).sort("timestamp").aggr("pid")
            template = "- %(pid)+6s %(user)-10s %(age)+3s  %(mem_avg)+5s " \
                    "(%(mem_min)s-%(mem_max)s)\n"
            for proc in procs.data:
                pid = proc["pid"]
                if proc["timestamp"][-1] != latest:
                    pid = "-" * len(str(pid))
                elif not os.path.isdir('/proc/%s' % pid):
                    pid = "dead"
                used_mem = proc["used_gpu_memory"]
                str_params = {"pid": pid,
                              "user": proc["user"][-1],
                              "age": Process.seconds_to_human(time.time() - proc["started"][-1]),
                              "mem_avg": int(sum(used_mem) / len(used_mem)),
                              "mem_min": ljust(min(used_mem), 5),
                              "mem_max": rjust(max(used_mem), 5)}
                bufs[-1] += template % str_params
            bufs.append("")
        return bufs[:-1]
