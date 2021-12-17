import os
from pathlib import Path

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
