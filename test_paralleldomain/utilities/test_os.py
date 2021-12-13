from paralleldomain.utilities.os import cpu_count, is_container, virtual_memory


def test_is_container():
    _ = is_container()


def test_cpu_count():
    _ = cpu_count()


def test_virtual_memory():
    v_memory = virtual_memory()
    assert v_memory.total >= v_memory.used
