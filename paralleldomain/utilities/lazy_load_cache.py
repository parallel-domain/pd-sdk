import collections
import logging
import os
from sys import getsizeof
from threading import Event, RLock
from typing import Any, Callable, Dict, Hashable, Tuple, TypeVar

import numpy as np
import psutil
from cachetools import Cache
from humanize import naturalsize

CachedItemType = TypeVar("CachedItemType")

logger = logging.getLogger(__name__)

SHOW_CACHE_LOGS = os.environ.get("SHOW_CACHE_LOGS", False)


class CacheFullException(Exception):
    ...


class CacheEmptyException(Exception):
    ...


class LazyLoadCache(Cache):
    """Least Recently Used (LRU) cache implementation."""

    _marker = object()

    def __init__(self, max_ram_usage_factor: float = 0.8):
        self.max_ram_usage_factor = max_ram_usage_factor
        self.maximum_allowed_space: int = int(psutil.virtual_memory().total * self.max_ram_usage_factor)
        logger.info(f"Initializing LazyLoadCache with a max_ram_usage_factor of {max_ram_usage_factor}.")
        logger.info(f"This leads to a total available space of {naturalsize(self.maximum_allowed_space)}.")
        self._key_load_locks: Dict[Hashable, Tuple[RLock, Event]] = dict()
        self._create_key_lock = RLock()
        Cache.__init__(self, maxsize=self.maximum_allowed_space, getsizeof=LazyLoadCache.getsizeof)
        self.__order = collections.OrderedDict()

    def get_item(self, key: Hashable, loader: Callable[[], CachedItemType]) -> CachedItemType:
        key_lock, wait_event = self._get_locks(key=key)
        with key_lock:
            if key not in self:
                if SHOW_CACHE_LOGS:
                    logger.debug(f"load key {key} to cache")
                value = loader()
                try:
                    self[key] = value
                except CacheFullException as e:
                    logger.warning(f"Cant store {key} in Cache since no more space is left! {str(e)}")
                wait_event.set()
                return value
            return self[key]

    def __missing__(self, key):
        raise KeyError(key)

    def __getitem__(self, key: Hashable, cache_getitem: Callable[[Cache, Hashable], Any] = Cache.__getitem__):
        value = cache_getitem(self, key)
        if key in self:  # __missing__ may not store item
            self.__update(key)
        return value

    def __setitem__(self, key: Hashable, value, cache_setitem=Cache.__setitem__):
        self._custom_set_item(key, value)
        self.__update(key)

    def _get_locks(self, key: Hashable) -> Tuple[RLock, Event]:
        if key not in self._key_load_locks:
            with self._create_key_lock:
                if key not in self._key_load_locks:
                    self._key_load_locks[key] = (RLock(), Event())
        return self._key_load_locks[key]

    def _custom_set_item(self, key, value):
        size = self.getsizeof(value)
        if SHOW_CACHE_LOGS:
            logger.debug(f"add item {key} with size {naturalsize(size)}")
        if size > self.maxsize:
            raise ValueError("value too large")
        if key not in self._Cache__data or self._Cache__size[key] < size:
            try:
                while size > self.free_space:
                    self.popitem()
            except CacheEmptyException:
                if size > self.free_space:
                    raise CacheFullException(f"Cache is already empty but there is no more space left tho store {key}!")

        if key in self._Cache__data:
            diffsize = size - self._Cache__size[key]
        else:
            diffsize = size

        self._Cache__data[key] = value
        self._Cache__size[key] = size
        self._Cache__currsize += diffsize

    def __delitem__(self, key: Hashable, cache_delitem=Cache.__delitem__):
        key_lock, wait_event = self._get_locks(key=key)

        with key_lock:
            if wait_event.is_set():
                if SHOW_CACHE_LOGS:
                    logger.debug(f"delete {key} from cache")
                cache_delitem(self, key)
                del self.__order[key]
                wait_event.clear()

    @property
    def maxsize(self):
        """The maximum size of the cache."""
        return psutil.virtual_memory().total

    @property
    def free_space(self) -> int:
        """The maximum size of the caches free space."""
        remaining_allowed_space = self.maximum_allowed_space - self._Cache__currsize
        free_space = int(max(0, min(psutil.virtual_memory().free, remaining_allowed_space)))

        if SHOW_CACHE_LOGS:
            logger.debug(f"current cache free space {naturalsize(free_space)}")
        return free_space

    def popitem(self):
        """Remove and return the `(key, value)` pair least recently used."""
        try:
            key = next(iter(self.__order))
        except StopIteration:
            raise CacheEmptyException("%s is empty" % type(self).__name__)
        else:
            del self[key]

    def pop(self, key, default=_marker):
        key_lock, wait_event = self._get_locks(key=key)
        with key_lock:
            if key in self:
                value = self[key]
                del self[key]
            elif default is LazyLoadCache._marker:
                raise KeyError(key)
            else:
                value = default
            return value

    def clear(self):
        "D.clear() -> None.  Remove all items from D."
        try:
            while True:
                self.popitem()
        except CacheEmptyException:
            pass

    def __update(self, key):
        try:
            self.__order.move_to_end(key)
        except KeyError:
            self.__order[key] = None

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
        elif isinstance(value, np.ndarray):
            size = value.nbytes
        else:
            pass
        return size


cache_max_ram_usage_factor = float(os.environ.get("CACHE_MAX_USAGE_FACTOR", 0.1))  # 10% free space max

LAZY_LOAD_CACHE = LazyLoadCache(max_ram_usage_factor=cache_max_ram_usage_factor)
