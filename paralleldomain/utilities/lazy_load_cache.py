import os
from sys import getsizeof
from typing import Any, Callable, Dict, Hashable, Type, TypeVar, Union

import cachetools

CachedItemType = TypeVar("CachedItemType")


class LazyLoadCache(cachetools.TTLCache):
    _type_to_size: Dict[Type, Union[int, Callable[[Any], int]]] = dict()

    def get_item(self, key: Hashable, loader: Callable[[], CachedItemType]) -> CachedItemType:
        if key not in self:
            self[key] = loader()
        return self[key]

    @staticmethod
    def getsizeof(value):
        """Return the size of a cache element's value."""
        return getsizeof(value)


_cache_max_size_in_bytes = os.environ.get("CACHE_MAX_SIZE_BYTES", 1e10)  # 10 GB default
_cache_max_time_to_live = os.environ.get("CACHE_TIME_TO_LIVE_SECONDS", 600)  # 10 mins default

LAZY_LOAD_CACHE = LazyLoadCache(maxsize=int(_cache_max_size_in_bytes), ttl=_cache_max_time_to_live)
