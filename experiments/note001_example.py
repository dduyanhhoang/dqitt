"""Numbers for the DQITT step-by-step note: Notion NOTE-001's 4-section / 3-teacher example run through DQI.

    uv run python experiments/note001_example.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dqitt import paths
from dqitt.circuit import DQICircuit, lookup_table, row_coeffs
from dqitt.dqi import all_x, expected_satisfied, ideal_state
from dqitt.problem import brute_force, build, code_distance, decoder_failure, load

OUT = Path("results/note001_example")


def main() -> None:
    inst = load(paths.DATA / "dqitt" / "note001")
    lin = build(inst)
    X = all_x(lin.p, lin.n)
    s = lin.count(X)
    opt, xs = brute_force(lin)
    lines = [
        f"eligible {[sorted(e) for e in inst.eligible]}, conflicts {inst.conflicts}",
        f"p={lin.p} n={lin.n} m={lin.m}",
        "B = " + str(lin.B.tolist()),
        "F = " + str([np.flatnonzero(f).tolist() for f in lin.F]),
        f"optimum {opt}/{lin.m} at {len(xs)} schedules: {[tuple(x) for x in xs.tolist()]}",
        f"code distance d = {code_distance(lin, max_w=4)}",
        f"eps_0..2 = {np.round(decoder_failure(lin, 2), 4).tolist()}",
        f"random <s>/m = {s.mean() / lin.m:.4f}, P(optimal) = {(s == opt).mean():.4f}",
        f"conflict row 0 c(e) = {np.round(row_coeffs(lin, 0), 4).tolist()}",
        f"eligibility row 3 c(e) = {np.round(row_coeffs(lin, 3), 4).tolist()}",
    ]
    for l in (1, 2):
        psi, _, w = ideal_state(lin, l)
        circ = DQICircuit(lin, l, w)
        pr = circ.probs_x()
        lines += [
            f"l={l}: w = {np.round(w, 4).tolist()}, LUT rows = {len(lookup_table(lin, l))}, qubits = {circ.n_wires}",
            f"l={l}: ideal <s>/m = {expected_satisfied(psi, lin, X) / lin.m:.4f}, P(optimal) = {(np.abs(psi) ** 2)[s == opt].sum():.4f}",
            f"l={l}: circuit <s>/m = {(pr * s).sum() / lin.m:.4f}, P(optimal) = {pr[s == opt].sum():.4f}, "
            f"circuit == ideal: {np.allclose(pr, np.abs(psi) ** 2, atol=1e-10)}",
        ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
