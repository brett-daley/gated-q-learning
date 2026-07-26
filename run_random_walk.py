# MIT License
# Copyright (c) 2026 Brett Daley

import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Any, Callable, ClassVar, Self, override

import numpy as np

from utils import math, parallel
from utils.interface import BasicEnvironment
from utils.key import Key
from utils.parallel import MapFuture
from utils.space import Space
from utils.stopwatch import stopwatch
from utils.typing import (
    Environment,
    EnvironmentLike,
    FrozenList,
    Make,
    Transition,
)

NUM_STEPS: int = 500
NUM_RUNS: int = 300


class RandomWalk(BasicEnvironment):
    NUM_STATES: ClassVar[int] = 19
    STATES: ClassVar[FrozenList[int]] = tuple(range(NUM_STATES))
    START_STATE: ClassVar[int] = NUM_STATES // 2
    TERMINAL_STATES: ClassVar[tuple[int, int]] = (-1, NUM_STATES)
    # Distinct rewards to make action gap large
    REWARDS: ClassVar[tuple[float, float]] = (-1.0, 1.0)
    OPTIMAL_ACTION: ClassVar[int] = 1

    @property
    def observation_space(self) -> Space:
        return Space.discrete(self.NUM_STATES)

    @property
    def action_space(self) -> Space:
        return Space.discrete(2)

    @override
    def reset(self) -> int:
        return self.START_STATE

    @override
    def _next_state(self, state: int, action: int) -> int:
        return {
            0: state - 1,
            1: state + 1,
        }[action]

    @override
    def _reward(self, state: int, action: int, next_state: int) -> float:
        if next_state == self.TERMINAL_STATES[0]:
            return self.REWARDS[0]
        if next_state == self.TERMINAL_STATES[1]:
            return self.REWARDS[1]
        return 0.0

    @override
    def _done(self, state: int, action: int, next_state: int) -> bool:
        return next_state in self.TERMINAL_STATES


DISCOUNT = 0.99


Q_STAR = np.zeros((RandomWalk.NUM_STATES, 2))
Q_STAR[0, 0] = RandomWalk.REWARDS[0]

states = np.arange(RandomWalk.NUM_STATES)
dist_right = (RandomWalk.NUM_STATES - 1) - states

Q_STAR[:, 1] = RandomWalk.REWARDS[1] * (DISCOUNT**dist_right)
Q_STAR[1:, 0] = DISCOUNT * Q_STAR[:-1, 1]


class RandomWalkAgent(ABC):
    def __init__(
        self, key: Key, env: EnvironmentLike, step_size: float, discount: float
    ) -> None:
        self._rng = np.random.default_rng(int(key))
        self.num_actions = env.action_space.count
        self.step_size = step_size
        self.discount = discount
        # Tiny noise to break initial ties
        self.q = self._rng.normal(
            size=[env.observation_space.count, self.num_actions],
            scale=1e-9,
        )
        self.visited = np.zeros_like(self.q[:, 0])

        self.init_error = math.rms_error(self.q, Q_STAR)

    @classmethod
    def make(cls, key: Key, env: EnvironmentLike, /, **kwargs: Any) -> Self:
        return cls(key, env, **kwargs)

    def select_action(self, obs: int) -> tuple[Self, int]:
        return self, self._rng.choice(self.num_actions)

    @abstractmethod
    def update(self, transition: Transition) -> Self:
        raise NotImplementedError()

    def greedy_policy(self) -> np.ndarray:
        return np.argmax(self.q, axis=1)

    def compute_accuracy(self) -> float:
        where_correct = self.greedy_policy() == RandomWalk.OPTIMAL_ACTION
        where_tied = self.q[:, 0] == self.q[:, 1]
        mask = self.visited
        count = np.sum(mask)
        if count == 0:
            return 0.5
        return np.sum(mask * np.logical_or(where_correct, where_tied)) / count

    def compute_error(self) -> float:
        error = math.rms_error(self.q, Q_STAR)
        return 1.0 - (error / self.init_error)


class GatedQLambda(RandomWalkAgent):
    @override
    def __init__(
        self,
        key: Key,
        env: EnvironmentLike,
        /,
        step_size: float,
        lambd: float,
        gate: float,
        discount: float = 1.0,
    ) -> None:
        super().__init__(key, env, step_size, discount)
        self.lambd = lambd
        self.gate = gate
        self.z = np.zeros_like(self.q)

    @override
    def update(self, transition: Transition) -> Self:
        state, action, next_state, reward, done = transition
        self.visited[state] = 1.0  # Update for accuracy computation

        chosen_q = self.q[state, action]
        max_q = np.max(self.q[state])
        is_greedy = chosen_q == max_q

        ql_error = reward - chosen_q
        td_error = reward - max_q
        if not done:
            bootstrap = self.discount * np.max(self.q[next_state])
            ql_error += bootstrap
            td_error += bootstrap

        decay = self.discount * self.lambd
        if not is_greedy:
            decay *= self.gate
        self.z *= decay

        self.q += self.step_size * td_error * self.z
        self.q[state, action] += self.step_size * ql_error

        if not done:
            self.z[state, action] += 1.0
        else:
            self.z *= 0.0

        return self


def run_experiment(
    make_env: Make[Environment],
    make_agent: Callable[[Key, EnvironmentLike], RandomWalkAgent],
    key: Key,
) -> list[float]:
    env = make_env(key)
    env, obs = env.observe()
    agent = make_agent(key, env)

    return_values: list[float] = [agent.compute_error()]
    for _ in range(NUM_STEPS):
        agent, action = agent.select_action(obs)
        env, (reward, done) = env.step(action)
        env, next_obs = env.observe()
        transition = Transition(obs, action, next_obs, reward, done)
        agent = agent.update(transition)
        obs = transition.next_obs

        return_values.append(agent.compute_error())

    return return_values


def run_experiments_in_parallel(
    make_env: Make[EnvironmentLike],
    make_agent: Callable[[Key, EnvironmentLike], RandomWalkAgent],
) -> MapFuture[FrozenList[FrozenList[float]]]:
    f: Make[FrozenList[list[float]]] = partial(run_experiment, make_env, make_agent)
    keys = [Key.create(seed) for seed in range(NUM_RUNS)]
    return parallel.submit_map(f, keys)


@stopwatch
def main(seed: int) -> None:
    N = 21
    GRID = np.linspace(0.0, 1.0, num=N)
    print(GRID)

    scores = np.empty(shape=[N, N, N, NUM_RUNS, NUM_STEPS + 1], dtype=float)
    futures = {}

    # Dispatch experiments
    for i, x in enumerate(GRID):  # alpha
        for j, y in enumerate(GRID):  # lambda
            for k, z in enumerate(GRID):  # gate
                futures[(i, j, k)] = run_experiments_in_parallel(
                    make_env=RandomWalk.make,
                    make_agent=partial(
                        GatedQLambda.make,
                        discount=DISCOUNT,
                        step_size=x,
                        lambd=y,
                        gate=z,
                    ),
                )

    print("Gathering results now...")

    # Gather results
    for i, x in enumerate(GRID):  # alpha
        for j, y in enumerate(GRID):  # lambda
            for k, z in enumerate(GRID):  # gate
                print(i, j, k, flush=True)
                scores[i, j, k] = futures[(i, j, k)].result()

    # Cache data for plotting later
    np.save("scores.npy", scores)


if __name__ == "__main__":
    main(0)
