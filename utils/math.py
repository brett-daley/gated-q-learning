# MIT License
# Copyright (c) 2026 Brett Daley

import numpy as np


def rms_error(a: np.ndarray, b: np.ndarray) -> float:
    assert a.shape == b.shape
    return np.sqrt(np.mean(np.square(a - b)))
