"""DQITT-005: <H> of ttqaoa's published p=1 angles vs the angles re-optimised by experiments/qaoa_rerun.py.

Both evaluated on the current ttqaoa Hamiltonian (inst_01..10). Writes results/qaoa_rerun/energy_check.csv.

    uv run --group baselines python experiments/qaoa_energy_check.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from loguru import logger

from dqitt import paths
from dqitt.ttqaoa import energies, qaoa_probs, qubo

sys.path.insert(0, str(Path(__file__).parent))
from runs_to_optimal import published_params, rerun_params  # noqa: E402

OUT = Path("results/qaoa_rerun/energy_check.csv")


def main() -> None:
    logger.remove()
    rows = []
    for i in range(1, 11):
        name = f"inst_{i:02d}"
        problem = qubo(paths.DATA / "ttqaoa" / "ensembles" / name)
        e = energies(problem)
        rows.append(dict(
            instance=name, min_energy=e.min(), mean_energy_uniform=e.mean(),
            published_p1=qaoa_probs(problem, published_params(name)[1]) @ e,
            rerun_p1=qaoa_probs(problem, rerun_params(name)) @ e,
        ))
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(OUT.read_text())


if __name__ == "__main__":
    main()
