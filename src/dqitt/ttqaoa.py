"""Adapter to the ttqaoa QAOA baseline (package qmos_qaoa, dependency group `baselines`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pennylane as qml
from qmos_qaoa.data import load_local_data
from qmos_qaoa.model import TimetableProblem


def qubo(data_dir: str | Path) -> TimetableProblem:
    return TimetableProblem(load_local_data(data_dir))


def energies(problem: TimetableProblem) -> np.ndarray:
    """Cost of every basis state; index i has wire 0 as its most significant bit (PennyLane order)."""
    h = qml.simplify(problem.cost_h)
    return np.real(h.sparse_matrix(wire_order=range(problem.n_qubits)).diagonal())


def index_to_bits(i: np.ndarray, n: int) -> np.ndarray:
    return (np.asarray(i)[..., None] >> np.arange(n - 1, -1, -1)) & 1


def bits_to_assignment(bits: np.ndarray, problem: TimetableProblem) -> tuple[np.ndarray, np.ndarray]:
    """Teacher per section for bitstrings (..., n_qubits), -1 = none; and whether each bitstring is a
    schedule at all (no section with two or more teachers)."""
    n_sec = len(problem.sections)
    onehot = np.zeros((*bits.shape[:-1], n_sec, len(problem.teachers)), int)
    for j, (t, s) in enumerate(problem.feasible_vars):
        onehot[..., s, t] = bits[..., j]
    count = onehot.sum(axis=-1)
    return np.where(count == 1, onehot.argmax(axis=-1), -1), (count <= 1).all(axis=-1)


def qaoa_probs(problem: TimetableProblem, params: np.ndarray) -> np.ndarray:
    """Exact output distribution of ttqaoa's QAOA circuit (qmos_qaoa.solver) at angles params, shape (depth, 2)."""
    n = problem.n_qubits
    h = qml.simplify(problem.cost_h)

    @qml.qnode(qml.device("lightning.qubit", wires=n))
    def circuit():
        for i in range(n):
            qml.Hadamard(wires=i)
        for gamma, beta in params:
            qml.templates.TrotterProduct(h, gamma, order=1, n=1)
            for i in range(n):
                qml.RX(2 * beta, wires=i)
        return qml.probs(wires=range(n))

    return np.asarray(circuit())
