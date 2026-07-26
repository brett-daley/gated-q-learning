# MIT License
# Copyright (c) 2026 Brett Daley

from collections.abc import Sequence
from typing import Self, Type

import equinox as eqx
import numpy as np

from utils.typing import FrozenList


class Space(eqx.Module):
    type Shape = FrozenList[int]
    type Bound = int | float | np.ndarray

    shape: Shape
    dtype: Type = eqx.field(static=True)
    low: np.ndarray
    high: np.ndarray
    is_discrete: bool

    @classmethod
    def discrete(cls, n: int) -> Self:
        return cls._create(shape=(), dtype=np.int32, low=0, high=n - 1, discrete=True)

    @classmethod
    def _create(
        cls, shape: Sequence[int], dtype: Type, low: Bound, high: Bound, discrete: bool
    ) -> Self:
        shape = tuple(shape)
        low = _to_numpy(low, shape, dtype)
        high = _to_numpy(high, shape, dtype)
        return cls(shape, dtype, low, high, discrete)

    @property
    def count(self) -> int:
        if not self.is_discrete:
            raise TypeError("Continuous space is uncountable")
        return np.prod(self.high - self.low + 1).item()


def _to_numpy(bound: Space.Bound, shape: Space.Shape, dtype: Type) -> np.ndarray:
    bound = np.asanyarray(bound)
    if not np.can_cast(bound.dtype, dtype, casting="same_kind"):
        raise TypeError(f"Cannot cast {bound.dtype} to {dtype}")
    return np.broadcast_to(
        bound.astype(dtype),
        shape,
    )
