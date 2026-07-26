# MIT License
# Copyright (c) 2026 Brett Daley

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    NamedTuple,
    Protocol,
    Self,
    runtime_checkable,
)

if TYPE_CHECKING:
    from utils.key import Key
    from utils.space import Space
else:
    Key = Any
    Space = Any

type FrozenList[T] = tuple[T, ...]
"""A type alias for an arbitrary-length tuple.

Tuples are superior to lists in production code as they prevent bugs due to accidental
mutations, but their need for ellipses makes their type annotations more verbose. Use
this alias as a convenience substitute instead.
"""

type State = Any
type Observation = Any
type Action = Any
type Reward = float
type Done = bool

type Feedback = tuple[Reward, Done]


class Transition(NamedTuple):
    obs: Observation
    action: Action
    next_obs: Observation
    reward: Reward
    done: Done


type Make[T] = Callable[[Key], T]
"""A Make is a pure function that takes a key and returns an object.

The use of Makes is strongly encouraged to ensure reproducibility in experiments.
"""


class EnvironmentLike(Protocol):
    @property
    def observation_space(self) -> Space: ...

    @property
    def action_space(self) -> Space: ...

    def reset(self, key: Key, /) -> Self: ...

    def observe(self) -> tuple[Self, Observation]: ...

    def step(self, action: Action) -> tuple[Self, Feedback]: ...


@runtime_checkable
class Environment(EnvironmentLike, Protocol):
    @classmethod
    def make(cls, key: Key, /) -> EnvironmentLike: ...


__all__ = [
    "Action",
    "Done",
    "Environment",
    "FrozenList",
    "Make",
    "Observation",
    "Reward",
    "State",
    "Transition",
]
