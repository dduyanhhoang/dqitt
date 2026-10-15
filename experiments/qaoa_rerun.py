"""DQITT-005: re-run ttqaoa's QAOA (qmos_qaoa.solver.solve_qaoa, unchanged) on all 20 ensembles.

Same settings as ttqaoa's params_by_depths.py (20 restarts, 20000 shots, COBYLA), depth p=1 only:
deeper circuits take > 10 min per restart on this machine. Writes the best angles and the
20000 measured bitstrings (as basis-state indices) per instance.

    OMP_NUM_THREADS=1 uv run --group baselines python experiments/qaoa_rerun.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from qmos_qaoa.solver import solve_qaoa

from dqitt import paths
from dqitt.ttqaoa import qubo

OUT = Path("results/qaoa_rerun")
DEPTH, SHOTS, RESTARTS = 1, 20000, 20


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted((paths.DATA / "ttqaoa" / "ensembles").glob("inst_*")):
        problem = qubo(d)
        samples, params = solve_qaoa(problem, depth=DEPTH, shots=SHOTS, restarts=RESTARTS)
        idx = np.asarray(samples) @ (1 << np.arange(problem.n_qubits - 1, -1, -1))
        np.save(OUT / f"{d.name}_p{DEPTH}_samples.npy", idx.astype(np.int32))
        rows += [dict(instance=d.name, depth=DEPTH, layer=i + 1, gamma=g, beta=b) for i, (g, b) in enumerate(params)]
        print(f"{d.name} done", flush=True)
    with (OUT / "params.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
