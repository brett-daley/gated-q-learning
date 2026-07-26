# MIT License
# Copyright (c) 2026 Brett Daley

import atexit
import multiprocessing
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Callable

from tqdm import tqdm

from utils.typing import FrozenList

_executor = ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn"))
atexit.register(_executor.shutdown)


class MapFuture[T]:
    def __init__(self, futures: Sequence[Future[T]]):
        self._futures: FrozenList[Future[T]] = tuple(futures)
        self._results: list[T] | None = None

    def result(self, *, unzip: bool = False) -> FrozenList[T]:
        if self._results is None:
            self._results = self._get_results()
        results = self._results
        if unzip and isinstance(results[0], Iterable):
            return tuple(zip(*results))
        return tuple(results)

    def _get_results(self) -> list[T]:
        results = []
        for f in tqdm(self._futures, total=len(self._futures)):
            results.append(f.result())
        return results


def submit_map[T, R](func: Callable[[T], R], values: Iterable[T], /) -> MapFuture[R]:
    return MapFuture([_executor.submit(func, v) for v in values])


__all__ = [
    "MapFuture",
    "submit_map",
]
