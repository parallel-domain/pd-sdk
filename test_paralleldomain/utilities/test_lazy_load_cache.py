import os
from multiprocessing.pool import ThreadPool
from random import random
from sys import getsizeof
from time import sleep
from unittest import mock
from unittest.mock import patch

import pytest

from paralleldomain.utilities.lazy_load_cache import CacheEmptyException, LazyLoadCache


def test_max_size():
    if "SKIP_CACHE" not in os.environ:
        with patch("psutil.virtual_memory") as mocked_virtual_memory:
            mocked_virtual_memory.return_value.free = 2 * getsizeof(mock.MagicMock()) + 1
            mocked_virtual_memory.return_value.total = 5 * getsizeof(mock.MagicMock())
            mocked_virtual_memory.return_value.used = (
                mocked_virtual_memory.return_value.total - mocked_virtual_memory.return_value.free
            )

            cache = LazyLoadCache(max_ram_usage_factor=1.0, ram_keep_free_factor=0.0)

            def _pop_fake():
                LazyLoadCache.popitem(cache)
                mocked_virtual_memory.return_value.free += getsizeof(mock.MagicMock())
                mocked_virtual_memory.return_value.used -= getsizeof(mock.MagicMock())

            cache.popitem = _pop_fake
            mock1_loader = mock.MagicMock()
            mock2_loader = mock.MagicMock()
            mock3_loader = mock.MagicMock()

            cache.get_item(key="key1", loader=mock1_loader.load)
            mock1_loader.load.assert_called_once()
            mocked_virtual_memory.return_value.free -= getsizeof(mock.MagicMock())
            mocked_virtual_memory.return_value.used += getsizeof(mock.MagicMock())

            cache.get_item(key="key2", loader=mock2_loader.load)
            mock2_loader.load.assert_called_once()
            mocked_virtual_memory.return_value.free -= getsizeof(mock.MagicMock())
            mocked_virtual_memory.return_value.used += getsizeof(mock.MagicMock())
            cache.get_item(key="key1", loader=mock1_loader.load)
            mock1_loader.load.assert_called_once()
            mock2_loader.load.assert_called_once()

            cache.get_item(key="key3", loader=mock3_loader.load)
            mock3_loader.load.assert_called_once()
            mocked_virtual_memory.return_value.free -= getsizeof(mock.MagicMock())
            mocked_virtual_memory.return_value.used += getsizeof(mock.MagicMock())
            assert "key2" not in cache
            assert "key3" in cache
            assert "key1" in cache

            mock3_loader.load.assert_called_once()
            cache.get_item(key="key2", loader=mock2_loader.load)
            assert mock2_loader.load.call_count == 2
            mocked_virtual_memory.return_value.free -= getsizeof(mock.MagicMock())
            mocked_virtual_memory.return_value.used += getsizeof(mock.MagicMock())
            assert "key1" not in cache
            assert "key3" in cache
            assert "key2" in cache

            cache.get_item(key="key4", loader=mock2_loader.load)
            assert mock2_loader.load.call_count == 3
            mocked_virtual_memory.return_value.free -= getsizeof(mock.MagicMock())
            mocked_virtual_memory.return_value.used += getsizeof(mock.MagicMock())
            assert "key1" not in cache
            assert "key3" not in cache
            assert "key4" in cache
            assert "key2" in cache


def test_thread_safe_delete():
    cache = LazyLoadCache(max_ram_usage_factor=0.2)
    cache.get_item(key="test", loader=mock.MagicMock)

    cache.pop(key="test", default=None)

    def delete(_):
        with pytest.raises(CacheEmptyException):
            cache.popitem()

    p1 = ThreadPool(140)
    p1.map_async(delete, range(140)).get()
