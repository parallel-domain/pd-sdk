import os
import sys
from dataclasses import dataclass
from pathlib import Path

import psutil
from cgroupspy.trees import Tree


def is_container() -> bool:
    """Returns `True` if process is running inside a (Docker) container."""
    cgroup_path = Path("/proc/self/cgroup")
    return (
        Path("/.dockerenv").exists()
        or cgroup_path.is_file()
        and any("docker" in line for line in cgroup_path.open(mode="r"))
    )


def cpu_count() -> int:
    """Returns CPU count. If process is running inside a (Docker) container,
    it returns the allocated virtual CPU count."""
    if is_container():
        cgroup_tree = Tree()
        cpu_controller = cgroup_tree.get_node_by_path("/cpu/").controller
        if cpu_controller.cfs_quota_us is None:  # running inside AWS-hosted container
            return int(cpu_controller.shares / 1024)
        else:  # running locally hosted container
            return int(cpu_controller.cfs_quota_us / cpu_controller.cfs_period_us)
    else:
        return os.cpu_count()


@dataclass
class Memory:
    total: int
    used: int

    @property
    def available(self) -> int:
        return self.total - self.used


def virtual_memory() -> Memory:
    """Returns total, used and available memory. If process is running inside a (Docker) container,
    it returns the allocated memory limit, or pyhsical host memory if no explicit limit has been set."""
    if is_container():
        cgroup_tree = Tree()
        memory_controller = cgroup_tree.get_node_by_path("/memory/").controller

        memory_limit = memory_controller.limit_in_bytes
        if memory_limit == (sys.maxsize - sys.maxsize % 4096):  # no container limit - take host physical limit
            memory_limit = psutil.virtual_memory().total
        return Memory(total=memory_limit, used=memory_controller.usage_in_bytes)
    else:
        psutil_vm = psutil.virtual_memory()
        return Memory(total=psutil_vm.total, used=psutil_vm.used)
