"""dqitt DATA_DIR [--l L] [--shots N] [--circuit] [--device D] [--out DIR]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from loguru import logger

from . import viz
from .dqi import all_x, expected_satisfied, ideal_state, sample, theory_weights
from .problem import brute_force, build, code_distance, decoder_failure, load


def main() -> None:
    ap = argparse.ArgumentParser(description="DQI on a timetable instance (max-LINSAT over F_p, no quota)")
    ap.add_argument("data_dir")
    ap.add_argument("--l", type=int, default=2, help="polynomial degree")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--no-eligibility", action="store_true", help="conflict rows only")
    ap.add_argument("--circuit", action="store_true", help="also run the PennyLane circuit (tiny instances only)")
    ap.add_argument("--device", default="lightning.qubit", help="pennylane device, e.g. lightning.gpu on HPC")
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    inst = load(a.data_dir)
    lin = build(inst, eligibility=not a.no_eligibility)
    opt, xs = brute_force(lin)
    d = code_distance(lin)
    logger.info(f"optimum {opt}/{lin.m} ({len(xs)} solutions); code distance {d}; unique decoding up to l={((d or 99) - 1) // 2}")
    logger.info(f"decoder failure eps_k (k=0..l): {np.round(decoder_failure(lin, a.l), 3)}")

    X = all_x(lin.p, lin.n)
    logger.info(f"random baseline <s>/m = {lin.count(X).mean() / lin.m:.4f}")
    r = lin.F.sum(axis=1)
    if (r == r[0]).all():
        logger.info(f"tridiagonal theory (needs 2l+1 < d) <s>/m = {theory_weights(lin.m, a.l, lin.p, int(r[0]))[1]:.4f}")
    psi, X, w = ideal_state(lin, a.l)
    logger.success(f"ideal DQI l={a.l}: <s>/m = {expected_satisfied(psi, lin, X) / lin.m:.4f}, w = {np.round(w, 3)}")

    samples = {"random": rng.integers(0, lin.p, (a.shots, lin.n)), f"DQI ideal l={a.l}": sample(psi, X, a.shots, rng)}
    if a.circuit:
        from .circuit import DQICircuit

        circ = DQICircuit(lin, a.l, w, device=a.device)
        logger.info(f"circuit: {circ.n_wires} wires, LUT {len(circ.table)} rows")
        pr = circ.probs_x()
        logger.success(f"circuit <s>/m = {(pr * lin.count(X)).sum() / lin.m:.4f}")
        samples["DQI circuit"] = X[rng.choice(len(pr), a.shots, p=pr / pr.sum())]
        viz.circuit(circ, a.out / "circuit.png")

    best = max(samples[f"DQI ideal l={a.l}"], key=lin.count)
    logger.info(f"best sampled assignment ({lin.count(best)}/{lin.m}): "
                + ", ".join(f"{s}->{inst.teachers[t] if t < len(inst.teachers) else 'dummy'}" for s, t in zip(inst.sections, best)))
    viz.satisfied_histogram(lin, samples, opt, a.out / "histogram.png")
    viz.conflict_graph(inst, a.out / "conflict_graph.png", assignment=best)
    logger.info(f"figures written to {a.out}/")
