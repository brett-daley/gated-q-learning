# MIT License
# Copyright (c) 2026 Brett Daley

import time
from functools import wraps
from typing import Any, Callable


def stopwatch(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Function `{func.__name__}` returned after {end - start:.3f} seconds")
        return result

    return wrapper
