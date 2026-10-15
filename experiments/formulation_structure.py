"""DQITT-003: structure of each formulation on the ttqaoa ensembles.

For every instance and formulation: variables n, rows m, field p, dual code distance d,
decoder failure eps_1/eps_2 (ideal min-weight decoder), qubits of the implemented DQI circuit,
and, for the ttqaoa QUBO, qubits and Hamiltonian terms.

    uv run --group baselines python experiments/formulation_structure.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pennylane as qml
from loguru import logger
from qmos_qaoa.data import load_local_data
from qmos_qaoa.model import TimetableProblem

from dqitt import paths
from dqitt.circuit import n_wires
from dqitt.formulations import binary
from dqitt.problem import build, code_distance, decoder_failure, load

OUT = Path("results/formulation_structure")
COVERS = {
    "teacher-index": "elig+conflict",
    "binary-none": "elig+conflict",
    "binary-hard": "all",
    "binary-soft": "all (soft quota)",
    "binary-hard-merged": "all",
}


def rows(d: Path) -> list[dict]:
    inst = load(d)
    lins = {"teacher-index": build(inst)} | {f"binary-{q}": binary(inst, q)[0] for q in ("none", "hard", "soft")}
    lins["binary-hard-merged"] = binary(inst, "hard", merge=True)[0]
    out = []
    for name, lin in lins.items():
        dist = code_distance(lin, max_w=3)
        eps = decoder_failure(lin, 2)
        out.append(dict(instance=d.name, formulation=name, covers=COVERS[name], n=lin.n, m=lin.m, p=lin.p,
                        d=dist, eps1=round(eps[1], 4), eps2=round(eps[2], 4), qubits=n_wires(lin)))
    qubo = TimetableProblem(load_local_data(d))
    terms = len(qml.simplify(qubo.cost_h).terms()[0])
    out.append(dict(instance=d.name, formulation="ttqaoa-qubo", covers="all (penalties)", n=qubo.n_qubits,
                    qubits=qubo.n_qubits, terms=terms))
    return out


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    OUT.mkdir(parents=True, exist_ok=True)
    table = [r for d in sorted((paths.DATA / "ttqaoa" / "ensembles").glob("inst_*")) for r in rows(d)]
    fields = ["instance", "formulation", "covers", "n", "m", "p", "d", "eps1", "eps2", "qubits", "terms"]
    with (OUT / "ensembles.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(table)
    print(f"wrote {OUT / 'ensembles.csv'} ({len(table)} rows)")


if __name__ == "__main__":
    main()
