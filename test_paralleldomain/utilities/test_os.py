from paralleldomain.utilities.os import cpu_count, is_container


def test_is_container():
    # Test execution to assure no error is thrown on non-unix (non-cgroup) OS
    is_container()


def test_cpu_count():
    # Test execution to assure no error is thrown on non-unix (non-cgroup) OS
    cpu_count()
