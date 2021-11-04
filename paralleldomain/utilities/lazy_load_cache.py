import collections
import logging
import os
import re
from sys import getsizeof
from threading import Event, RLock
from typing import Any, Callable, Dict, Hashable, Tuple, TypeVar, Union

import numpy as np
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

    def __init__(self, cache_name: str = "Default pd-sdk Cache", cache_max_size: str = "1GiB"):
        self.cache_name = cache_name
        self._maximum_allowed_bytes: int = byte_str_to_bytes(byte_str=cache_max_size)
        logger.info(
            f"Initializing LazyLoadCache '{cache_name}' with available "
            f"space of {naturalsize(self._maximum_allowed_bytes)}."
        )

        self._key_load_locks: Dict[Hashable, Tuple[RLock, Event]] = dict()
        self._create_key_lock = RLock()
        Cache.__init__(self, maxsize=self._maximum_allowed_bytes, getsizeof=LazyLoadCache.getsizeof)
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
                    return value
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
            self.free_space_for_n_bytes(n_bytes=size)

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

    def free_space_for_n_bytes(self, n_bytes: Union[float, int]):
        try:
            while n_bytes > self.free_space:
                self.popitem()
        except CacheEmptyException:
            if n_bytes > self.free_space:
                raise CacheFullException(
                    f"Cache is already empty but there is no more space left tho store {n_bytes}B!"
                )

    @property
    def maxsize(self) -> int:
        """The maximum size of the cache."""
        return self._maximum_allowed_bytes

    @maxsize.setter
    def maxsize(self, value: Union[str, int]):
        if isinstance(value, int):
            self._maximum_allowed_bytes = value
        elif isinstance(value, str):
            self._maximum_allowed_bytes: int = byte_str_to_bytes(byte_str=value)
        else:
            raise ValueError(f"invalid type for maxsite {type(value)}! Has to be int or str.")
        logger.info(f"Changed '{self.cache_name}' available space to {naturalsize(self._maximum_allowed_bytes)}.")
        # If size got smaller make sure cache is cleared up
        self.free_space_for_n_bytes(n_bytes=0)

    @property
    def currsize(self) -> int:
        """The current size of the cache."""
        return int(self._Cache__currsize)

    @property
    def free_space(self) -> int:
        """The maximum size of the caches free space."""
        remaining_allowed_space = self.maxsize - self.currsize
        return remaining_allowed_space

    def popitem(self):
        """Remove and return the `(key, value)` pair least recently used."""
        try:
            it = iter(list(self.__order.keys()))
            key = next(it)
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
                size += LazyLoadCache.getsizeof(i)
        elif isinstance(value, dict):
            for k, v in value.items():
                size += LazyLoadCache.getsizeof(v)
        elif isinstance(value, np.ndarray):
            size = value.nbytes
        return size


def byte_str_to_bytes(byte_str: str) -> int:
    split_numbers_and_letters = re.match(r"([0-9]+)([kKMGTPEZY]*)([i]*)([bB]+)", byte_str.replace(" ", ""), re.I)
    powers = {"": 0, "k": 1, "m": 2, "g": 3, "t": 4, "p": 5, "e": 6, "z": 7, "y": 8}
    if split_numbers_and_letters is None:
        raise ValueError(f"Invalid byte string format {byte_str}. Has to be a int number followed by a byte unit!")
    number, power_letter, base_letter, bites_or_bytes = split_numbers_and_letters.groups()
    bit_factor = 1 if bites_or_bytes == "B" else 1 / 8
    base = 1024 if base_letter == "i" else 1000
    power = powers[power_letter.lower()]
    number = float(number)
    total_bits = number * base ** power * bit_factor
    return int(total_bits)


cache_max_ram_usage_factor = float(os.environ.get("CACHE_MAX_USAGE_FACTOR", 0.1))  # 10% free space max
cache_max_size = os.environ.get("CACHE_MAX_BYTES", "1GiB")
if "CACHE_MAX_USAGE_FACTOR" in os.environ:
    logger.warning(
        "CACHE_MAX_USAGE_FACTOR is not longer supported! Use CACHE_MAX_BYTES instead to set a cache size in bytes!"
    )

LAZY_LOAD_CACHE = LazyLoadCache(cache_max_size=cache_max_size)
