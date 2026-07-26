# MIT License
# Copyright (c) 2026 Brett Daley

from typing import Self

import equinox as eqx
import jax


class Key(eqx.Module):
    """An immutable key for pseudorandom number generation."""

    _key: jax.Array

    @classmethod
    def create(cls, seed: int) -> Self:
        return cls(jax.random.PRNGKey(seed))

    def __int__(self) -> int:
        v0, v1 = map(int, jax.random.key_data(self._key))
        return int(v0) | (int(v1) << 32)
