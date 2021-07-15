import collections
import os
from sys import getsizeof
from threading import RLock
from typing import Any, Callable, Dict, Hashable, Set, Type, TypeVar, Union

import psutil
from cachetools import Cache

CachedItemType = TypeVar("CachedItemType")


class LazyLoadCache(Cache):
    _delete_lock = RLock()
    """Least Recently Used (LRU) cache implementation."""

    def __init__(self, max_ram_usage_factor: float = 0.8):
        self.max_ram_usage_factor = max_ram_usage_factor
        self.maximum_allowed_space: int = int(psutil.virtual_memory().total * self.max_ram_usage_factor)
        self._lock_prefixes: Set[str] = set()
        self._key_load_locks: Dict[Hashable, RLock] = dict()
        Cache.__init__(self, maxsize=self.maximum_allowed_space, getsizeof=LazyLoadCache.getsizeof)
        self.__order = collections.OrderedDict()

    def get_item(self, key: Hashable, loader: Callable[[], CachedItemType]) -> CachedItemType:
        with LazyLoadCache._delete_lock:
            has_key = key in self
            if not has_key and key not in self._key_load_locks:
                self._key_load_locks[key] = RLock()

        with self._key_load_locks[key]:
            if key not in self:
                print(f"load {key}")
                self[key] = loader()
            return self[key]

    def __getitem__(self, key: Hashable, cache_getitem=Cache.__getitem__):
        with LazyLoadCache._delete_lock:
            value = cache_getitem(self, key)
            if key in self:  # __missing__ may not store item
                self.__update(key)
            return value

    def __setitem__(self, key: Hashable, value, cache_setitem=Cache.__setitem__):
        with LazyLoadCache._delete_lock:
            self._custom_set_item(key, value)
            self.__update(key)

    def _custom_set_item(self, key, value):
        size = self.getsizeof(value)
        if size > self.maxsize:
            raise ValueError("value too large")
        if key not in self._Cache__data or self._Cache__size[key] < size:
            while size > self.free_space:
                popped_item = self.popitem()
                if popped_item is None:
                    print(f"we can't find anything to delete in cache, so we just add {key} anyways")
                    break  # we can't find anything to delete in cache, so we just add it anyways

        if key in self._Cache__data:
            diffsize = size - self._Cache__size[key]
        else:
            diffsize = size

        self._Cache__data[key] = value
        self._Cache__size[key] = size
        self._Cache__currsize += diffsize

    def __delitem__(self, key: Hashable, cache_delitem=Cache.__delitem__):
        with LazyLoadCache._delete_lock:
            print(f"delete {key}")
            cache_delitem(self, key)
            del self.__order[key]

    @property
    def maxsize(self):
        """The maximum size of the cache."""
        return psutil.virtual_memory().total

    @property
    def free_space(self) -> int:
        """The maximum size of the caches free space."""
        remaining_allowed_space = self.maximum_allowed_space - self._Cache__currsize
        return int(max(0, min(psutil.virtual_memory().free, remaining_allowed_space)))

    def popitem(self):
        """Remove and return the `(key, value)` pair least recently used."""
        # try:
        #     key = next(iter(self.__order))
        # except StopIteration:
        #     raise KeyError("%s is empty" % type(self).__name__) from None
        # else:
        #     return (key, self.pop(key))

        found_key_to_remove = False
        num_locked_items = 0
        with LazyLoadCache._delete_lock:
            key_iter = iter(self.__order)
            while not found_key_to_remove:
                try:
                    key = next(key_iter)
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

    def __update(self, key):
        try:
            self.__order.move_to_end(key)
            print(f"moved {key} to back")
        except KeyError:
            self.__order[key] = None

    def _is_locked_key(self, key: str):
        return any([key.startswith(locked) for locked in self._lock_prefixes])

    def clear_prefix(self, prefix: str):
        for key in self:
            if key.startswith(prefix):
                self.pop(key=key)

    def lock_prefix(self, prefix: str):
        self._lock_prefixes.add(prefix)

    def unlock_prefix(self, prefix: str):
        if prefix in self._lock_prefixes:
            self._lock_prefixes.remove(prefix)

    @staticmethod
    def getsizeof(value):
        """Return the size of a cache element's value."""
        size = getsizeof(value)
        if hasattr(value, "__dict__"):
            pass
            # for k, v in value.__dict__.items():
            #     size += getsizeof(v)
        elif isinstance(value, list):
            for i in value:
                size += getsizeof(i)
        elif isinstance(value, dict):
            for k, v in value.items():
                size += getsizeof(v)
        else:
            pass
        return size


_cache_max_ram_usage_factor = float(os.environ.get("CACHE_MAX_USAGE_FACOTR", 0.5))  # 50% free space max

LAZY_LOAD_CACHE = LazyLoadCache(max_ram_usage_factor=_cache_max_ram_usage_factor)
