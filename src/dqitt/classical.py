"""Classical references for the original problem (eligibility, one teacher per section, conflict, quota).

The objective everywhere is `problem.violations`: assignments are in {-1, 0..k-1}^n
(-1 = no teacher), and an optimal schedule is one with the fewest broken constraints.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from ortools.sat.python import cp_model

from .problem import Instance, violations


def optimum(inst: Instance) -> tuple[int, np.ndarray]:
    """Fewest violations over all (k+1)^n assignments, and every assignment reaching it.

    ponytail: full enumeration, fine for the 8-section ensembles (5^8 rows); use cpsat_optimum beyond that.
    """
    k, n = len(inst.teachers), len(inst.sections)
    A = np.array(list(product(range(-1, k), repeat=n)))
    v = violations(inst, A)
    return int(v.min()), A[v == v.min()]


def _solve(model: cp_model.CpModel, time_limit: float) -> tuple[cp_model.CpSolver, str]:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    return solver, solver.status_name(solver.Solve(model))


def cpsat_optimum(inst: Instance, time_limit: float = 60.0) -> tuple[int, np.ndarray]:
    """CP-SAT model of the same objective: minimise the number of violated constraints."""
    n, k = len(inst.sections), len(inst.teachers)
    m = cp_model.CpModel()
    x = {(s, t): m.NewBoolVar(f"x_{s}_{t}") for s in range(n) for t in range(k)}
    none = [m.NewBoolVar(f"none_{s}") for s in range(n)]
    cost = list(none)
    for s in range(n):
        m.AddExactlyOne([x[s, t] for t in range(k)] + [none[s]])
        cost += [x[s, t] for t in range(k) if t not in inst.eligible[s]]
    for s, u in inst.conflicts:
        for t in range(k):
            clash = m.NewBoolVar(f"clash_{s}_{u}_{t}")
            m.Add(clash >= x[s, t] + x[u, t] - 1)
            cost.append(clash)
    for t, (lo, hi) in enumerate(inst.quota or []):
        broken = m.NewBoolVar(f"quota_{t}")
        load = sum(x[s, t] for s in range(n))
        m.Add(load >= lo).OnlyEnforceIf(broken.Not())
        m.Add(load <= hi).OnlyEnforceIf(broken.Not())
        cost.append(broken)
    m.Minimize(sum(cost))
    solver, status = _solve(m, time_limit)
    if status != "OPTIMAL":
        raise RuntimeError(f"CP-SAT stopped with {status}")
    a = np.array([next((t for t in range(k) if solver.Value(x[s, t])), -1) for s in range(n)])
    return int(solver.ObjectiveValue()), a


def cpsat_ttqaoa_notebook(
    inst: Instance, w_section: float = 300, w_conflict: float = 300, time_limit: float = 60.0
) -> tuple[str, np.ndarray | None]:
    """The CP-SAT model of ttqaoa notebooks/share/QAOA_Full.ipynb, cell 18 (commit 3b494c1).

    Kept as written: variables only for eligible pairs, section term W^2(2*sum z - sum x),
    conflict term W^2(x1 + x2 + 2z) (the expansion of the QUBO's (x1 + x2)^2), quota as hard
    constraints. The notebook builds the objective but never calls Minimize or Solve; both are
    added here. Sections without exactly one chosen teacher come back as -1.
    """
    n, k = len(inst.sections), len(inst.teachers)
    pairs = [(t, s) for t in range(k) for s in range(n) if t in inst.eligible[s]]  # ttqaoa variable order
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x_{j}") for j in range(len(pairs))]

    def product_var(i: int, j: int) -> cp_model.IntVar:
        z = m.NewBoolVar(f"z_{i}_{j}")
        m.Add(z <= x[i])
        m.Add(z <= x[j])
        m.Add(z >= x[i] + x[j] - 1)
        return z

    objective = []
    for s in range(n):
        idx = [j for j, (_, ss) in enumerate(pairs) if ss == s]
        if idx:
            z = [product_var(i, j) for a, i in enumerate(idx) for j in idx[a + 1 :]]
            objective.append(w_section**2 * (2 * sum(z) - sum(x[i] for i in idx)))
    conflicting = {frozenset(c) for c in inst.conflicts}
    for t in range(k):
        idx = [(s, j) for j, (tt, s) in enumerate(pairs) if tt == t]
        for a, (s1, i) in enumerate(idx):
            for s2, j in idx[a + 1 :]:
                if frozenset((s1, s2)) in conflicting:
                    objective.append(w_conflict**2 * (x[i] + x[j] + 2 * product_var(i, j)))
    for t, (lo, hi) in enumerate(inst.quota or []):
        mine = [x[j] for j, (tt, _) in enumerate(pairs) if tt == t]
        if mine:
            m.Add(sum(mine) >= lo)
            m.Add(sum(mine) <= hi)
    m.Minimize(sum(objective))
    solver, status = _solve(m, time_limit)
    if status not in ("OPTIMAL", "FEASIBLE"):
        return status, None
    chosen = [[t for j, (t, ss) in enumerate(pairs) if ss == s and solver.Value(x[j])] for s in range(n)]
    return status, np.array([c[0] if len(c) == 1 else -1 for c in chosen])
