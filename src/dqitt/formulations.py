"""Binary encoding x_{s,l} as max-LINSAT over F_p (teammate's proposal, Notion DQITT-001).

One variable per eligible (section s, teacher l) pair, so eligibility needs no row. Rows:
  domain    x_j                   in {0, 1}
  one-hot   sum_l x_{s,l}         in {1}
  conflict  x_{s,l} + x_{s',l}    in {0, 1}        per teacher l eligible for both sides of a conflict
  quota     sum_s x_{s,l}         hard: in {Qmin..Qmax};  soft: one row per k = 0..deg(l) with F = {k}
p is the smallest prime above every row's term count, so sums of 0/1 values never wrap mod p.
The soft variant's per-k score r_{l,k} cannot enter the DQI state (rows are unweighted);
only the row structure is built here.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .problem import Instance, Linsat, next_prime


def binary(
    inst: Instance, quota: Literal["hard", "soft", "none"] = "hard", merge: bool = False
) -> tuple[Linsat, list[tuple[int, int]]]:
    """Returns the LINSAT instance and the (section, teacher) pair behind each column.

    Rows with equal coefficients form a weight-2 dual codeword (d = 2, so DQI cannot correct
    even one error). merge=True replaces each such group by one row whose allowed set is the
    intersection: the set of schedules satisfying every row is unchanged, the row count is not.
    """
    pairs = [(s, l) for s, E in enumerate(inst.eligible) for l in sorted(E)]
    col = {pr: j for j, pr in enumerate(pairs)}
    rows: list[tuple[list[int], set[int]]] = [([j], {0, 1}) for j in range(len(pairs))]
    rows += [([col[s, l] for l in sorted(E)], {1}) for s, E in enumerate(inst.eligible) if E]
    rows += [([col[s, l], col[u, l]], {0, 1}) for s, u in inst.conflicts for l in sorted(inst.eligible[s] & inst.eligible[u])]
    load = [[col[s, l] for s, E in enumerate(inst.eligible) if l in E] for l in range(len(inst.teachers))]
    if quota == "hard":
        rows += [(c, set(range(lo, hi + 1))) for c, (lo, hi) in zip(load, inst.quota) if c]
    elif quota == "soft":
        rows += [(c, {k}) for c in load if c for k in range(len(c) + 1)]
    if merge:
        if quota == "soft":
            raise ValueError("soft quota rows share coefficients by design; merging would erase them")
        merged: dict[tuple[int, ...], set[int]] = {}
        for c, vals in rows:
            merged[tuple(c)] = merged.get(tuple(c), vals) & vals
        rows = [(list(c), vals) for c, vals in merged.items() if vals]  # empty = never satisfiable, constant

    p =next_prime(max(len(c) for c, _ in rows) + 1)
    B = np.zeros((len(rows), len(pairs)), int)
    for i, (c, _) in enumerate(rows):
        B[i, c] = 1
    F = np.array([np.isin(np.arange(p), list(vals)) for _, vals in rows])
    return Linsat(p, B, F), pairs


def to_assignment(x: np.ndarray, pairs: list[tuple[int, int]], n_sections: int) -> np.ndarray | None:
    """Teacher per section (-1 = none), or None if x is not a 0/1 schedule (a value outside {0, 1},
    or a section with two or more teachers)."""
    if not np.isin(x, (0, 1)).all():
        return None
    a = np.full(n_sections, -1)
    count = np.zeros(n_sections, int)
    for (s, l), v in zip(pairs, x):
        if v:
            a[s], count[s] = l, count[s] + 1
    return a if (count <= 1).all() else None
