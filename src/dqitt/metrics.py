"""How many runs (shots) a sampler needs to return an optimal schedule at least once."""

from __future__ import annotations

import math

import numpy as np


def shots_to_hit(p_opt: float, confidence: float) -> float:
    """Smallest N with P(at least one optimal sample in N shots) = 1 - (1 - p_opt)^N >= confidence."""
    if p_opt <= 0:
        return math.inf
    if p_opt >= 1:
        return 1
    return math.ceil(math.log(1 - confidence) / math.log(1 - p_opt))


def first_hit(is_optimal: np.ndarray) -> int | None:
    """1-based index of the first optimal shot in a sequence, None if there is none."""
    hits = np.flatnonzero(is_optimal)
    return int(hits[0]) + 1 if hits.size else None
