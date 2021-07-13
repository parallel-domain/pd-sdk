import os
from sys import getsizeof
from threading import Lock
from typing import Any, Callable, Dict, Hashable, Set, Type, TypeVar, Union

import cachetools
import psutil
from cachetools import Cache, TTLCache

CachedItemType = TypeVar("CachedItemType")


class LazyLoadCache(TTLCache):
    _delete_lock = Lock()
    _type_to_size: Dict[Type, Union[int, Callable[[Any], int]]] = dict()

    def __init__(self, ttl: int, max_ram_usage_factor: float = 0.8):
        self.max_ram_usage_factor = max_ram_usage_factor
        self._lock_prefixes: Set[str] = set()
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

    def clear_prefix(self, prefix: str):
        for key in self:
            if key.startswith(prefix):
                self.pop(key=key)

    def lock_prefix(self, prefix: str):
        self._lock_prefixes.add(prefix)

    def unlock_prefix(self, prefix: str):
        if prefix in self._lock_prefixes:
            self._lock_prefixes.remove(prefix)
            self.expire()

    def expire(self, time=None):
        """Remove expired items from the cache."""
        with LazyLoadCache._delete_lock:
            if time is None:
                time = self._TTLCache__timer()
            root = self._TTLCache__root
            curr = root.next
            links = self._TTLCache__links
            cache_delitem = Cache.__delitem__
            while curr is not root and curr.expire < time:
                if curr.key is not None and not self._is_locked_key(key=curr.key):
                    cache_delitem(self, curr.key)
                    del links[curr.key]
                    next = curr.next
                    curr.unlink()
                    curr = next
                else:
                    curr = curr.next

    def _is_locked_key(self, key: str):
        return any([key.startswith(locked) for locked in self._lock_prefixes])

    def popitem(self):
        """Remove and return the `(key, value)` pair least recently used that
        has not already expired.

        """
        with self._TTLCache__timer as time:
            self.expire(time)
            found_key_to_remove = False
            num_locked_items = 0
            with LazyLoadCache._delete_lock:
                while not found_key_to_remove:
                    try:
                        key = next(iter(self._TTLCache__links))
                        is_locked = self._is_locked_key(key=key)
                        found_key_to_remove = not is_locked
                        if is_locked:
                            num_locked_items += 1
                            continue
                    except StopIteration:
                        if num_locked_items == 0:
                            raise KeyError("%s is empty" % type(self).__name__) from None
                        return None
                    else:
                        return (key, self.pop(key))


class LazyLoadingMixin:
    def __init__(self, unique_cache_key_prefix: str):
        self._unique_cache_key_prefix = unique_cache_key_prefix

    def clear_cached_items(self):
        LAZY_LOAD_CACHE.clear_prefix(prefix=self._unique_cache_key_prefix)


_cache_max_ram_usage_factor = float(os.environ.get("CACHE_MAX_USAGE_FACOTR", 0.8))  # 80% free space max
_cache_max_time_to_live = int(os.environ.get("CACHE_TIME_TO_LIVE_SECONDS", 600))  # 10 mins default

LAZY_LOAD_CACHE = LazyLoadCache(ttl=_cache_max_time_to_live, max_ram_usage_factor=_cache_max_ram_usage_factor)
