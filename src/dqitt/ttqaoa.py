"""Adapter to the ttqaoa QAOA baseline (package qmos_qaoa, dependency group `baselines`)."""

from __future__ import annotations

from copy import copy
from pathlib import Path

import numpy as np
import pennylane as qml
from qmos_qaoa.data import load_local_data
from qmos_qaoa.model import TimetableProblem


def qubo(data_dir: str | Path, **weights: float) -> TimetableProblem:
    """ttqaoa's QUBO; weights are TimetableProblem's w_section, w_conflict, w_quota (defaults 3, 2, 1)."""
    return TimetableProblem(load_local_data(data_dir), **weights)


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


def exact_qubo(problem: TimetableProblem) -> qml.operation.Operator:
    """QUBO whose energy on a schedule is its number of broken eligibility/conflict constraints (no quota).

    sum_s (sum_t x_ts - 1)^2 + sum_{t, conflicting s < s'} x_ts x_ts'. Unlike ttqaoa's (x_ts + x_ts')^2
    conflict term, a conflict costs nothing unless both sections get teacher t.
    """

    def proj(i: int) -> qml.operation.Operator:
        return (1 - qml.PauliZ(i)) / 2

    terms = []
    for s in problem.sections:
        idx = [j for j, (_, ss) in enumerate(problem.feasible_vars) if ss == s]
        if idx:
            terms.append((sum(proj(i) for i in idx) - 1) ** 2)
    conflict = problem.slot_conflict
    for t in range(len(problem.teachers)):
        mine = [(problem.task_list[s]["Slot"], j) for j, (tt, s) in enumerate(problem.feasible_vars) if tt == t]
        for a, (slot1, i) in enumerate(mine):
            for slot2, j in mine[a + 1 :]:
                if slot1 in conflict.index and slot2 in conflict.columns and conflict.at[slot1, slot2] == 1:
                    terms.append(proj(i) @ proj(j))
    return qml.simplify(sum(terms))


def with_cost(problem: TimetableProblem, cost_h: qml.operation.Operator) -> TimetableProblem:
    """Copy of problem with another cost Hamiltonian (same variables), usable by solve_qaoa and qaoa_probs."""
    other = copy(problem)
    other.cost_h = cost_h
    return other
