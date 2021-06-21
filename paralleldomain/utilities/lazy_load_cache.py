import os
from sys import getsizeof
from typing import Any, Callable, Dict, Hashable, Type, TypeVar, Union

import cachetools
import psutil

CachedItemType = TypeVar("CachedItemType")


class LazyLoadCache(cachetools.TTLCache):
    _type_to_size: Dict[Type, Union[int, Callable[[Any], int]]] = dict()

    def __init__(self, ttl: int, max_ram_usage_factor: float = 0.8):
        self.max_ram_usage_factor = max_ram_usage_factor
        super().__init__(maxsize=self.free_space, ttl=ttl)

    def get_item(self, key: Hashable, loader: Callable[[], CachedItemType]) -> CachedItemType:
        if key not in self:
            self[key] = loader()
        return self[key]

    @staticmethod
    def getsizeof(value):
        """Return the size of a cache element's value."""
        return getsizeof(value)

    def __setitem__(self, key, value, **kwargs):
        super().__setitem__(key=key, value=value, cache_setitem=LazyLoadCache._custom_set_item)

    def _custom_set_item(self, key, value):
        size = self.getsizeof(value)
        if size > self.maxsize:
            raise ValueError("value too large")
        if key not in self._Cache__data or self._Cache__size[key] < size:
            while size > self.free_space:
                self.popitem()
        if key in self._Cache__data:
            diffsize = size - self._Cache__size[key]
        else:
            diffsize = size
        self._Cache__data[key] = value
        self._Cache__size[key] = size
        self._Cache__currsize += diffsize

    @property
    def maxsize(self):
        """The maximum size of the cache."""
        return psutil.virtual_memory().total

    @property
    def free_space(self) -> int:
        """The maximum size of the caches free space."""
        return max(0, psutil.virtual_memory().free, psutil.virtual_memory().total * (1.0 - self.max_ram_usage_factor))


_cache_max_ram_usage_factor = float(os.environ.get("CACHE_MAX_USAGE_FACOTR", 0.8))  # 80% free space max
_cache_max_time_to_live = int(os.environ.get("CACHE_TIME_TO_LIVE_SECONDS", 600))  # 10 mins default

LAZY_LOAD_CACHE = LazyLoadCache(ttl=_cache_max_time_to_live, max_ram_usage_factor=_cache_max_ram_usage_factor)
