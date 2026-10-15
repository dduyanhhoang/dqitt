"""Numbers for the step-by-step DQI note: the PennyLane tutorial's max-XORSAT instance run through dqitt.

Instance (pennylane.ai/qml/demos/tutorial_dqi): rows 1000, 1100, 0110, 0011, 0001, v = 10101, l = 2.

    uv run python experiments/tutorial_example.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dqitt.circuit import DQICircuit, lookup_table, row_coeffs
from dqitt.dqi import all_x, expected_satisfied, ideal_state, theory_weights
from dqitt.problem import Linsat, brute_force, code_distance, decoder_failure

OUT = Path("results/tutorial_example")
B = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
V = np.array([1, 0, 1, 0, 1])
L = 2


def main() -> None:
    lin = Linsat(2, B, np.array([[v == 0, v == 1] for v in V]))
    X = all_x(lin.p, lin.n)
    s = lin.count(X)
    opt, xs = brute_force(lin)
    w_th, frac_th = theory_weights(lin.m, L, lin.p, 1)
    psi, _, w = ideal_state(lin, L)
    psi_th, _, _ = ideal_state(lin, L, w=w_th)
    circ = DQICircuit(lin, L, w)
    pr = circ.probs_x()
    lines = [
        f"m={lin.m} n={lin.n} p={lin.p} l={L}",
        f"optimum {opt}/{lin.m}, optimal x: {[''.join(map(str, x)) for x in xs]}",
        f"code distance d = {code_distance(lin, max_w=5)}",
        f"decoder failure eps_0..2 = {np.round(decoder_failure(lin, L), 4).tolist()}",
        f"lookup table rows = {len(lookup_table(lin, L))}",
        f"row 0 Fourier coefficients c_0(e) = {np.round(row_coeffs(lin, 0), 4).tolist()}",
        f"tridiagonal weights w = {np.round(w_th, 4).tolist()}, predicted <s>/m = {frac_th:.4f}",
        f"exact weights w = {np.round(w, 4).tolist()}, exact <s>/m = {expected_satisfied(psi, lin, X) / lin.m:.4f}",
        f"tutorial weights in the exact state: <s>/m = {expected_satisfied(psi_th, lin, X) / lin.m:.4f}, "
        f"P(optimal) = {(np.abs(psi_th) ** 2)[s == opt].sum():.4f}",
        f"random <s>/m = {s.mean() / lin.m:.4f}",
        f"P(optimal): random {(s == opt).mean():.4f}, ideal DQI {(np.abs(psi) ** 2)[s == opt].sum():.4f}, "
        f"circuit {pr[s == opt].sum():.4f}",
        f"circuit qubits = {circ.n_wires}, circuit == ideal: {np.allclose(pr, np.abs(psi) ** 2, atol=1e-10)}",
        "P(x) per x (ideal): " + ", ".join(f"{''.join(map(str, x))}:{q:.4f}" for x, q in zip(X, np.abs(psi) ** 2)),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
