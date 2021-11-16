from multiprocessing.pool import ThreadPool
from unittest import mock

import pytest

from paralleldomain.utilities.lazy_load_cache import CacheEmptyException, LazyLoadCache


class _MockSizeElement:
    def __init__(self, desired_size: int):
        self.desired_size = desired_size

    def __sizeof__(self):
        return self.desired_size

    @staticmethod
    def get_mocked_loader(desired_size: int) -> mock.MagicMock:
        mock_loader = mock.MagicMock()
        element = _MockSizeElement(desired_size=desired_size)
        mock_loader.load.return_value = element
        return mock_loader


class TestLazyLoadCache:
    def test_removes_items_on_exceeding_max_size(self):
        cache = LazyLoadCache(cache_max_size="1200B")

        mock1_loader = _MockSizeElement.get_mocked_loader(desired_size=500)
        mock2_loader = _MockSizeElement.get_mocked_loader(desired_size=500)
        mock3_loader = _MockSizeElement.get_mocked_loader(desired_size=500)
        mock4_loader = _MockSizeElement.get_mocked_loader(desired_size=1000)

        cache.get_item(key="key1", loader=mock1_loader.load)
        mock1_loader.load.assert_called_once()
        assert len(cache.keys()) == 1
        assert "key1" in cache
        cache.get_item(key="key2", loader=mock2_loader.load)
        mock2_loader.load.assert_called_once()
        assert len(cache.keys()) == 2
        assert "key2" in cache
        assert "key1" in cache

        # key1 and key2 should be present in cache. So no reloading needed
        cache.get_item(key="key1", loader=mock1_loader.load)
        mock1_loader.load.assert_called_once()
        mock2_loader.load.assert_called_once()
        assert len(cache.keys()) == 2
        assert "key2" in cache
        assert "key1" in cache

        # loading key3 should remove key2 since its least recently used
        cache.get_item(key="key3", loader=mock3_loader.load)
        mock3_loader.load.assert_called_once()
        assert len(cache.keys()) == 2
        assert "key2" not in cache
        assert "key3" in cache
        assert "key1" in cache

        # key1 and key3 should be in cache now so key1 should be removed in favor of key2
        cache.get_item(key="key2", loader=mock2_loader.load)
        assert mock2_loader.load.call_count == 2
        assert len(cache.keys()) == 2
        assert "key1" not in cache
        assert "key3" in cache
        assert "key2" in cache

        # there should only be room for key4 so everything else should be removed to make space
        cache.get_item(key="key4", loader=mock4_loader.load)
        mock4_loader.load.assert_called_once()
        assert len(cache.keys()) == 1
        assert "key1" not in cache
        assert "key3" not in cache
        assert "key2" not in cache
        assert "key4" in cache

    def test_max_change_reduction_drops_values(self):
        cache = LazyLoadCache(cache_max_size="2700B")
        mock1_loader = _MockSizeElement.get_mocked_loader(desired_size=500)
        mock2_loader = _MockSizeElement.get_mocked_loader(desired_size=500)
        mock3_loader = _MockSizeElement.get_mocked_loader(desired_size=500)
        mock4_loader = _MockSizeElement.get_mocked_loader(desired_size=1000)
        cache.get_item(key="key1", loader=mock1_loader.load)
        cache.get_item(key="key2", loader=mock2_loader.load)
        cache.get_item(key="key3", loader=mock3_loader.load)
        cache.get_item(key="key4", loader=mock4_loader.load)
        assert len(cache.keys()) == 4
        assert "key1" in cache
        assert "key3" in cache
        assert "key2" in cache
        assert "key4" in cache
        # should only leave space for key3 and key4
        cache.maxsize = 1700
        assert len(cache.keys()) == 2
        assert "key1" not in cache
        assert "key3" in cache
        assert "key2" not in cache
        assert "key4" in cache
        # should only leave space for key4
        cache.maxsize = "1100B"
        assert len(cache.keys()) == 1
        assert "key1" not in cache
        assert "key3" not in cache
        assert "key2" not in cache
        assert "key4" in cache

    def test_max_change_increase_keeps_values(self):
        cache = LazyLoadCache(cache_max_size="2700B")
        mock1_loader = _MockSizeElement.get_mocked_loader(desired_size=500)
        mock2_loader = _MockSizeElement.get_mocked_loader(desired_size=500)
        mock3_loader = _MockSizeElement.get_mocked_loader(desired_size=500)
        mock4_loader = _MockSizeElement.get_mocked_loader(desired_size=1000)
        cache.get_item(key="key1", loader=mock1_loader.load)
        cache.get_item(key="key2", loader=mock2_loader.load)
        cache.get_item(key="key3", loader=mock3_loader.load)
        cache.get_item(key="key4", loader=mock4_loader.load)
        assert len(cache.keys()) == 4
        assert "key1" in cache
        assert "key3" in cache
        assert "key2" in cache
        assert "key4" in cache
        # should only leave space for key3 and key4
        cache.maxsize = 3500
        assert len(cache.keys()) == 4
        assert "key1" in cache
        assert "key3" in cache
        assert "key2" in cache
        assert "key4" in cache
        # should only leave space for key4
        cache.maxsize = "1GB"
        assert len(cache.keys()) == 4
        assert "key1" in cache
        assert "key3" in cache
        assert "key2" in cache
        assert "key4" in cache

    def test_thread_safe_delete(self):
        cache = LazyLoadCache(cache_max_size="50KB")
        cache.get_item(key="test", loader=mock.MagicMock)

        cache.pop(key="test", default=None)

        def delete(_):
            with pytest.raises(CacheEmptyException):
                cache.popitem()

        p1 = ThreadPool(140)
        p1.map_async(delete, range(140)).get()
