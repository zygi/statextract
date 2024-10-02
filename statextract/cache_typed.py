
from perscache.serializers import CloudPickleSerializer, Serializer
from perscache.storage import CacheExpired, LocalFileStorage, Storage
from perscache.cache import _CachedFunction, hash_it, is_async
import perscache
from perscache._logger import debug, trace

from beartype import beartype
from beartype.typing import Any, Callable, Iterable, Optional, ParamSpec, TypeVar
from icontract import require

import datetime as dt
import functools
import hashlib
import inspect

import cloudpickle


Param = ParamSpec("Param")
RetType = TypeVar("RetType")

class Cache(perscache.Cache):
    """A cache that can be used to memoize functions."""

    @beartype
    def __init__(self, serializer: Serializer = None, storage: Storage = None):
        """Initialize the cache.

        Args:
            serializer: The serializer to use. If not specified, CloudPickleSerializer is used.
            storage: The storage to use. If not specified, LocalFileStorage is used.
        """

        self.serializer = serializer or CloudPickleSerializer()
        self.storage = storage or LocalFileStorage()

    def __repr__(self) -> str:
        return f"<Cache(serializer={self.serializer}, storage={self.storage})>"

    @beartype
    @require(
        lambda ttl: ttl is None or ttl > dt.timedelta(seconds=0),
        "ttl must be positive.",
    )
    def __call__(
        self,
        fn: Optional[Callable[Param, RetType]] = None,
        *,
        ignore: Optional[Iterable[str]] = None,
        serializer: Optional[Serializer] = None,
        storage: Optional[Storage] = None,
        ttl: Optional[dt.timedelta] = None,
    ) -> Callable[Param, RetType]:
        """Cache the value of the wrapped function.

        Tries to find a cached result of the decorated function in persistent storage.
        Returns the saved result if it was found, or calls the decorated function
        and caches its result.

        The cache will be invalidated if the function code, its argument values or
        the cache serializer have been changed.

        Args:
            ignore: A list of argument names to ignore when hashing the function.
            serializer: The serializer to use. If not specified, the default serializer is used.
                    Defaults to None.
            storage: The storage to use. If not specified, the default storage is used.
                    Defaults to None.
            ttl: The expiration time of the cache. If None, the cache will never expire.
                    Defaults to None.
        """

        if isinstance(ignore, str):
            ignore = [ignore]

        wrapper = _CachedFunction(
            self, ignore, serializer or self.serializer, storage or self.storage, ttl
        )

        # The decorator should work both with and without parentheses
        return wrapper if fn is None else wrapper(fn)

    cache = __call__  # Alias for backwards compatibility.

    @staticmethod
    @trace
    def _get(
        key: str, serializer: Serializer, storage: Storage, deadline: dt.datetime
    ) -> Any:
        data = storage.read(key, deadline)
        return serializer.loads(data)

    @staticmethod
    @trace
    def _set(key: str, value: Any, serializer: Serializer, storage: Storage) -> None:
        data = serializer.dumps(value)
        storage.write(key, data)

    @staticmethod
    def _get_hash(
        fn: Callable,
        args: tuple,
        kwargs: dict,
        serializer: Serializer,
        ignore: Iterable[str],
    ) -> str:

        # Remove ignored arguments from the arguments tuple and kwargs dict
        arg_dict = inspect.signature(fn).bind(*args, **kwargs).arguments

        if ignore is not None:
            arg_dict = {k: v for k, v in arg_dict.items() if k not in ignore}

        return hash_it(inspect.getsource(fn), type(serializer).__name__, arg_dict)

    def _get_filename(self, fn: Callable, key: str, serializer: Serializer) -> str:
        return f"{fn.__name__}-{key}.{serializer.extension}"
