"""
Module for caching function values.
"""

# TODO:
#  - replace tvA, tvB with _A, _B
#  - use Generics with ArgSet? TypedDict for kwargs
#  - Add docstring explaining hashing behaviour
#  - Profile to see if the ArgSet.__hash__ is a sufficiently efficient solution
#    If not, make a litecaching module that doesn't support kwargs.

from __future__ import annotations

from functools import wraps
from typing import (
    Mapping, Tuple, NamedTuple, Callable, List, Dict,
    TypeVar, Generic, Any, Optional, Type
)

import numpy as np


tvA = TypeVar("tvA")
tvB = TypeVar("tvB")
tvC = TypeVar("tvC", bound=Callable)

Decorator = Callable[[Callable], Callable]
NumpyArray = np.ndarray


_ArgSet__salt = hash("ArgSet")


class ArgSet(NamedTuple):
    """
    Stores sets of arguments for a function.
    """

    args: Tuple
    kwargs: Mapping[str, Any]

    def __eq__(self, other) -> bool:
        return (
            type(self) == type(other) and  # short-circuit
            self.args == other.args and self.kwargs == other.kwargs
        )

    def __hash__(self) -> int:
        return hash((_ArgSet__salt, self.args, tuple(self.kwargs.items())))


class Bounds(NamedTuple):

    upper: float
    lower: float


class StoredValue(Generic[tvB]):
    """
    Stores a cached value along with its flag.
    """

    value: tvB
    flag: bool

    def __init__(self, value: tvB, flag: bool = False):

        self.value = value
        self.flag = flag


class ContinuousCache(Generic[tvA]):
    """
    Caches functions of one float in a rectangular region to a given precision.

    Stores an array of cached values based on the given precision. Values that
    fall in between array indices are linearly interpolated.

    Uses numpy.
    """

    _dx: float
    _bounds: Bounds
    _xs: NumpyArray

    def __init__(self, bounds: Tuple[float, float], dx: float):

        bounds = Bounds(*bounds
)
        self._dx = dx
        self._bounds = bounds

        self._xs = np.arange(bounds.lower, bounds.upper + dx, dx)


class FunctionCache(Generic[tvA]):
    """
    Caches one function.

    Usage:
      - @FunctionCache()
        def myfunc(): ...

      - The cached function can be changed at any moment:
        fc = FunctionCache()
        @fc
        def f1(): ...

        @fc
        def f2(): ...
    """

    _returns: Dict[ArgSet, StoredValue[tvA]]
    _originalfunc: Callable

    def __init__(self):

        self._returns = {}

    # Use as decorator
    def __call__(self, func: Callable) -> Callable:

        return self.set_function(func)

    def reflag(self, *argsets: ArgSet) -> None:

        for argset in argsets:
            self._returns[argset].flag = False

    def reflagall(self) -> None:

        self.reflag(*(argset for argset in self._returns.keys()))

    def recalc(self, argset: ArgSet):

        stvalue = self._returns[argset]
        stvalue.value = self._originalfunc(*argset.args, **argset.kwargs)
        stvalue.flag = True

    def stored(self, argset: ArgSet) -> StoredValue:

        return self._returns[argset]

    # Decorator
    def set_function(self, func: tvC) -> tvC:
        """Decorator to set function."""

        self._originalfunc = func  # type: ignore

        @wraps(func)
        def decorated(*args, **kwargs) -> Any:

            # The meat of the caching!

            # Look for the stored return value in the dictionary.
            # If the corresponding ArgSet

            stvalue = self._returns.setdefault(
                ArgSet(args, kwargs),
                StoredValue(func(*args, **kwargs), True)
            )

            if stvalue.flag:
                return stvalue.value
            else:
                stvalue.value = func(*args, **kwargs)
                stvalue.flag = True
                return stvalue.value

        return decorated  # type: ignore


class ClassCache:

    _funccaches: List[FunctionCache]

    def __init__(self) -> None:

        self._funccaches = []

    def addfunction(self, tag: str) -> Decorator:

        # Create a new FunctionCache and add it to the ClassCache
        newfunccache: FunctionCache = FunctionCache()
        self._funccaches.append(newfunccache)

        # Return the funccache's decorator
        return newfunccache.set_function

    def recaches(self, tag: str) -> Decorator:

        ...
