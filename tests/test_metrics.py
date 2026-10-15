import math

import numpy as np

from dqitt.metrics import first_hit, shots_to_hit


def test_shots_to_hit_is_geometric_quantile():
    assert shots_to_hit(0.5, 0.95) == 5  # 1 - 0.5^5 = 0.969 >= 0.95 > 1 - 0.5^4
    assert shots_to_hit(1.0, 0.95) == 1
    assert shots_to_hit(0.0, 0.95) == math.inf


def test_first_hit_is_one_based():
    assert first_hit(np.array([False, False, True, True])) == 3
    assert first_hit(np.zeros(4, bool)) is None
