# MIT License
# Copyright (c) 2026 Brett Daley

from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from utils.key import Key
from utils.typing import Action, Done, Feedback, Observation, Reward, State


class BasicEnvironment(ABC):
    def __init__(self, key: Key) -> None:
        self._rng = np.random.default_rng(int(key))
        self._state = self.reset()

    @classmethod
    def make(cls, key: Key) -> Self:
        return cls(key)

    @abstractmethod
    def reset(self) -> State:
        raise NotImplementedError()

    def observe(self) -> tuple[Self, Observation]:
        return self, self._state

    def step(self, action: Action) -> tuple[Self, Feedback]:
        state = self._state
        next_state = self._next_state(state, action)
        reward = self._reward(state, action, next_state)
        done = self._done(state, action, next_state)
        self._state = self.reset() if done else next_state
        return self, (reward, done)

    @abstractmethod
    def _next_state(self, state: State, action: Action) -> State:
        raise NotImplementedError()

    @abstractmethod
    def _reward(self, state: State, action: Action, next_state: State) -> Reward:
        raise NotImplementedError()

    @abstractmethod
    def _done(self, state: State, action: Action, next_state: State) -> Done:
        raise NotImplementedError()
