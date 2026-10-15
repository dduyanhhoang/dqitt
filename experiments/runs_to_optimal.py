"""DQITT-005: qubits and runs-to-optimal for every method on the ttqaoa ensembles.

An outcome counts as optimal when it decodes to a schedule with V = V* broken constraints
(classical.optimum). Each method is an exact output distribution over its own outcome space:

  random-eligible        uniform over eligible teachers per section (classical sanity baseline)
  random-qubo            uniform over the 2^16 QUBO bitstrings
  random-teacher-index   uniform over F_5^8
  qaoa-published-p{d}    ttqaoa QAOA circuit at ttqaoa's published angles (inst_01..10, d = 1..5)
  qaoa-rerun-p1          ttqaoa QAOA re-optimised here (experiments/qaoa_rerun.py), plus its real shots
  dqi-teacher-index-l1   DQI on SPEC-001 (no quota), l=1: equals the circuit since eps_1 = 0
  dqi-teacher-index-l2   same, l=2, ideal decoder (the real lookup table fails on eps_2 of errors)
  dqi-binary-hard-l1     DQI on the merged binary hard-quota formulation, l=1 (closed form)

    uv run --group baselines python experiments/runs_to_optimal.py
"""

from __future__ import annotations

import csv
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
from loguru import logger

from dqitt import paths
from dqitt.circuit import n_wires
from dqitt.classical import optimum
from dqitt.dqi import all_x, ideal_state, l1_probs, l1_weights
from dqitt.formulations import binary, to_assignment
from dqitt.metrics import first_hit, shots_to_hit
from dqitt.problem import build, decoder_failure, load, violations
from dqitt.ttqaoa import bits_to_assignment, index_to_bits, qaoa_probs, qubo

OUT = Path("results/runs_to_optimal")
PUBLISHED = paths.DATA / "ttqaoa" / "processed" / "vikstal_params_ensemble.csv"
RERUN = Path("results/qaoa_rerun")
SHOTS = 20000


def published_params(name: str) -> dict[int, np.ndarray]:
    rows = [r for r in csv.DictReader(PUBLISHED.open()) if r["Instance"] == name]
    depths = sorted({int(r["Depth (p)"]) for r in rows})
    return {
        d: np.array([[float(r["Gamma (raw)"]), float(r["Beta (raw)"])] for r in rows if int(r["Depth (p)"]) == d])
        for d in depths
    }


def rerun_params(name: str) -> np.ndarray | None:
    path = RERUN / "params.csv"
    if not path.exists():
        return None
    rows = [r for r in csv.DictReader(path.open()) if r["instance"] == name]
    return np.array([[float(r["gamma"]), float(r["beta"])] for r in rows]) if rows else None


def score(method: str, qubits: int, prob: np.ndarray, optimal: np.ndarray, valid: np.ndarray, v: np.ndarray,
          rng: np.random.Generator, shots_idx: np.ndarray | None = None) -> dict:
    """Metrics of one exact distribution `prob` over outcomes with flags optimal/valid and violations v."""
    p_opt = float(prob[optimal].sum())
    if shots_idx is None:
        shots_idx = rng.choice(len(prob), SHOTS, p=prob / prob.sum())
    hits = optimal[shots_idx]
    p_valid = float(prob[valid].sum())
    return dict(
        method=method, qubits=qubits, p_opt=p_opt, p_valid=p_valid,
        mean_v_valid=float(prob[valid] @ v[valid] / p_valid) if p_valid else math.nan,
        expected_runs=1 / p_opt if p_opt else math.inf,
        n50=shots_to_hit(p_opt, 0.5), n95=shots_to_hit(p_opt, 0.95),
        first_hit=first_hit(hits), hits=int(hits.sum()),
    )


def rows(d: Path, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    inst = load(d)
    k = len(inst.teachers)
    v_star, _ = optimum(inst)
    out = []

    # eligible-only uniform
    E = [sorted(e) for e in inst.eligible]
    A = np.array(list(product(*E)))
    v = violations(inst, A)
    out.append(score("random-eligible", 0, np.full(len(A), 1 / len(A)), v == v_star, np.ones(len(A), bool), v, rng))

    # QUBO bitstrings: random and QAOA
    problem = qubo(d)
    n = problem.n_qubits
    a, valid = bits_to_assignment(index_to_bits(np.arange(2**n), problem.n_qubits), problem)
    v = violations(inst, a)
    opt = valid & (v == v_star)
    out.append(score("random-qubo", 0, np.full(2**n, 2.0**-n), opt, valid, v, rng))
    for depth, params in published_params(d.name).items():
        out.append(score(f"qaoa-published-p{depth}", n, qaoa_probs(problem, params), opt, valid, v, rng))
    params = rerun_params(d.name)
    if params is not None:
        shots = np.load(RERUN / f"{d.name}_p1_samples.npy")
        out.append(score("qaoa-rerun-p1", n, qaoa_probs(problem, params), opt, valid, v, rng, shots_idx=shots))

    # teacher-index F_p^n: random and DQI
    lin = build(inst)
    X = all_x(lin.p, lin.n)
    a = np.where(X < k, X, -1)
    v = violations(inst, a)
    every = np.ones(len(X), bool)
    out.append(score("random-teacher-index", 0, np.full(len(X), 1 / len(X)), v == v_star, every, v, rng))
    for l in (1, 2):
        psi, _, _ = ideal_state(lin, l)
        out.append(score(f"dqi-teacher-index-l{l}", n_wires(lin), np.abs(psi) ** 2, v == v_star, every, v, rng))

    # merged binary hard quota, l=1 closed form over the 0/1 corner of F_p^n
    lin, pairs = binary(inst, "hard", merge=True)
    assert decoder_failure(lin, 1)[1] == 0
    Xb = np.array(list(product((0, 1), repeat=lin.n)))
    decoded = [to_assignment(x, pairs, len(inst.sections)) for x in Xb]
    valid = np.array([x is not None for x in decoded])
    a = np.array([x if x is not None else np.full(len(inst.sections), -1) for x in decoded])
    v = violations(inst, a)
    prob = l1_probs(lin, l1_weights(lin), Xb)
    # mass outside {0,1}^n is never a schedule: one extra invalid outcome holds it
    prob = np.append(prob, max(0.0, 1 - prob.sum()))
    opt, valid, v = np.append(valid & (v == v_star), False), np.append(valid, False), np.append(v, 0)
    out.append(score("dqi-binary-hard-l1", n_wires(lin), prob, opt, valid, v, rng))
    return [dict(instance=d.name, v_star=v_star) | r for r in out]


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    OUT.mkdir(parents=True, exist_ok=True)
    dirs = sorted((paths.DATA / "ttqaoa" / "ensembles").glob("inst_*"))
    table = []
    for i, d in enumerate(dirs):
        table += rows(d, seed=i)
        print(f"{d.name} done", flush=True)
    with (OUT / "ensembles.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(f"wrote {OUT / 'ensembles.csv'} ({len(table)} rows)")


if __name__ == "__main__":
    main()
