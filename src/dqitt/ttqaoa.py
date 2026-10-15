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


def bits_to_assignment(bits: np.ndarray, problem: TimetableProblem) -> np.ndarray:
    """Teacher per section for bitstrings (..., n_qubits); -1 where a section has no or several teachers."""
    n_sec = len(problem.sections)
    onehot = np.zeros((*bits.shape[:-1], n_sec, len(problem.teachers)), int)
    for j, (t, s) in enumerate(problem.feasible_vars):
        onehot[..., s, t] = bits[..., j]
    return np.where(onehot.sum(axis=-1) == 1, onehot.argmax(axis=-1), -1)
