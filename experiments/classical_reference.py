"""DQITT-004: classical references on the ttqaoa ensembles.

Per instance: exact optimum V* (fewest violated constraints) and how many schedules reach it,
CP-SAT agreement and time, the ttqaoa notebook CP-SAT model's answer, and whether the ttqaoa
QUBO ground state is an optimal schedule.

    uv run --group baselines python experiments/classical_reference.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

from dqitt import paths
from dqitt.classical import cpsat_optimum, cpsat_ttqaoa_notebook, optimum
from dqitt.problem import load, violations
from dqitt.ttqaoa import bits_to_assignment, energies, index_to_bits, qubo

OUT = Path("results/classical_reference")


def row(d: Path) -> dict:
    inst = load(d)
    v_star, best = optimum(inst)
    t0 = time.perf_counter()
    v_cp, a_cp = cpsat_optimum(inst)
    t_cp = time.perf_counter() - t0
    status, a_nb = cpsat_ttqaoa_notebook(inst)
    problem = qubo(d)
    e = energies(problem)
    ground = np.flatnonzero(np.isclose(e, e.min()))
    a_ground, _ = bits_to_assignment(index_to_bits(ground, problem.n_qubits), problem)
    v_ground = violations(inst, a_ground)
    return dict(
        instance=d.name, v_star=v_star, n_optimal=len(best), feasible=v_star == 0,
        cpsat_v=v_cp, cpsat_agrees=v_cp == v_star == violations(inst, a_cp), cpsat_seconds=round(t_cp, 3),
        notebook_status=status, notebook_v=None if a_nb is None else violations(inst, a_nb),
        qubo_ground_states=len(ground), qubo_ground_v_min=int(v_ground.min()), qubo_ground_v_max=int(v_ground.max()),
    )


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    OUT.mkdir(parents=True, exist_ok=True)
    table = [row(d) for d in sorted((paths.DATA / "ttqaoa" / "ensembles").glob("inst_*"))]
    with (OUT / "ensembles.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(f"wrote {OUT / 'ensembles.csv'} ({len(table)} rows)")


if __name__ == "__main__":
    main()
