import time
from sys import getsizeof
from unittest import mock

from paralleldomain.utilities.lazy_load_cache import LazyLoadCache


def test_max_size():
    cache = LazyLoadCache(maxsize=2 * getsizeof(mock.MagicMock()), ttl=1555555)
    mock1_loader = mock.MagicMock()
    mock2_loader = mock.MagicMock()
    mock3_loader = mock.MagicMock()

    cache.get_item(key="key1", loader=mock1_loader.load)
    cache.get_item(key="key2", loader=mock2_loader.load)
    cache.get_item(key="key1", loader=mock1_loader.load)

    mock1_loader.load.assert_called_once()
    mock2_loader.load.assert_called_once()
    cache.get_item(key="key3", loader=mock3_loader.load)
    mock3_loader.load.assert_called_once()
    cache.get_item(key="key2", loader=mock2_loader.load)
    assert mock2_loader.load.call_count == 2


def test_max_size_with_mixed_types():
    load_dict_1 = dict(a=2)
    load_dict_2 = dict(a=1, b=2, c=3, d=4, e=6, f=22, g=87, h=123, i=44)
    load_dict_1_size = getsizeof(load_dict_1)
    load_dict_2_size = getsizeof(load_dict_2)
    mock_size = getsizeof(mock.MagicMock())

    cache = LazyLoadCache(maxsize=load_dict_2_size, ttl=1555555)
    mock1_loader = mock.MagicMock()
    mock2_loader = mock.MagicMock()
    mock2_loader.load.return_value = load_dict_1
    mock3_loader = mock.MagicMock()
    mock3_loader.load.return_value = load_dict_2

    cache.get_item(key="key1", loader=mock1_loader.load)
    assert cache.currsize == mock_size
    mock1_loader.load.assert_called_once()

    cache.get_item(key="key2", loader=mock2_loader.load)
    assert cache.currsize == mock_size + load_dict_1_size
    mock2_loader.load.assert_called_once()

    cache.get_item(key="key3", loader=mock3_loader.load)
    assert cache.currsize == load_dict_2_size
    assert "key1" not in cache
    assert "key2" not in cache

    assert mock1_loader.load.call_count == 1
    cache.get_item(key="key1", loader=mock1_loader.load)
    assert cache.currsize == mock_size
    assert mock1_loader.load.call_count == 2


def test_max_time():
    cache = LazyLoadCache(maxsize=100, ttl=2)
    mock1_loader = mock.MagicMock()

    cache.get_item(key="key1", loader=mock1_loader.load)
    assert cache.currsize == getsizeof(mock.MagicMock())
    mock1_loader.load.assert_called_once()
    time.sleep(2)
    assert cache.currsize == 0