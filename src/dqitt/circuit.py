"""DQI circuit over F_p (PennyLane). Each F_p symbol uses q = ceil(log2 p) qubits.

Registers: flags (m) | error (m symbols) | syndrome (n symbols). Stages (STAGES, run in order):
  dicke     flags <- sum_k w_k |Dicke_{m,k}>
  load      flag_i controls error_i <- sum_{e!=0} c_i(e) |e> (c_i = Fourier coeffs of g~_i), then flags uncomputed
  syndrome  syndrome <- B^T error   (controlled modular additions)
  decode    lookup table syndrome -> min-weight error, subtracted from the error register
  qft       QFT_p on every syndrome symbol; measuring gives x
Only feasible for tiny instances (m + q(m+n) qubits); the LUT has sum_k C(m,k)(p-1)^k rows.
"""

from __future__ import annotations

from math import ceil, comb, log2

import numpy as np
import pennylane as qml

from .problem import Linsat, error_patterns


def _perm(p: int, q: int, f) -> np.ndarray:
    """Unitary permuting basis states of F_p x F_p (padded to 2^{2q}) by (e, s) -> f(e, s)."""
    D = 2**q
    U = np.eye(D * D)
    for e in range(p):
        for s in range(p):
            src, dst = e * D + s, e * D + f(e, s)
            U[src, src] = 0
            U[dst, src] = 1
    return U


def _shift(p: int, q: int, c: int) -> np.ndarray:
    """|v> -> |v + c mod p> on one symbol (identity on padding states)."""
    U = np.eye(2**q)
    for v in range(p):
        U[v, v], U[(v + c) % p, v] = 0, 1
    return U


def qft_p(p: int, q: int) -> np.ndarray:
    U = np.eye(2**q, dtype=complex)
    w = np.exp(2j * np.pi / p)
    U[:p, :p] = w ** np.outer(np.arange(p), np.arange(p)) / np.sqrt(p)
    return U


def _local_prep(coeffs: np.ndarray, q: int) -> np.ndarray:
    """Householder unitary sending |0> to sum_e coeffs[e] |e> (coeffs[0] == 0)."""
    v = np.zeros(2**q, dtype=complex)
    v[: len(coeffs)] = coeffs
    u = np.eye(2**q)[0] - v
    return np.eye(2**q) - np.outer(u, u.conj())


def row_coeffs(lin: Linsat, i: int) -> np.ndarray:
    """c_i(e) = (1/p) sum_y g~_i(y) w^{-e y}; sum_{e!=0} |c_i|^2 = 1."""
    p = lin.p
    mean = lin.F[i].mean()
    g = (lin.F[i] - mean) / np.sqrt(mean * (1 - mean))
    w = np.exp(-2j * np.pi / p)
    return np.array([(g * w ** (e * np.arange(p))).sum() / p for e in range(p)])


def dicke_mixture(m: int, w: np.ndarray) -> np.ndarray:
    amp = np.zeros(2**m)
    for b in range(2**m):
        k = bin(b).count("1")
        if k < len(w):
            amp[b] = w[k] / np.sqrt(comb(m, k))
    return amp


def lookup_table(lin: Linsat, l: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(syndrome, error) pairs; min-weight error per syndrome, first wins on ties."""
    table: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
    for k in range(1, l + 1):
        for e in error_patterns(lin.m, lin.p, k):
            s = (e @ lin.B) % lin.p
            table.setdefault(s.tobytes(), (s, e))
    return list(table.values())


def n_wires(lin: Linsat) -> int:
    """Qubits of DQICircuit: m flags + ceil(log2 p) per error (m) and syndrome (n) symbol."""
    return lin.m + ceil(log2(lin.p)) * (lin.m + lin.n)


STAGES = ("dicke", "load", "syndrome", "decode", "qft")


def _bits(v: int, q: int) -> list[int]:
    return [int(b) for b in format(v, f"0{q}b")]


class DQICircuit:
    def __init__(self, lin: Linsat, l: int, w: np.ndarray, device: str = "default.qubit", shots: int | None = None):
        self.lin, self.l, self.w = lin, l, w
        self.q = ceil(log2(lin.p))
        m, n, q = lin.m, lin.n, self.q
        self.flags = list(range(m))
        self.err = [list(range(m + i * q, m + (i + 1) * q)) for i in range(m)]
        base = m + m * q
        self.syn = [list(range(base + j * q, base + (j + 1) * q)) for j in range(n)]
        self.n_wires = n_wires(lin)
        self.table = lookup_table(lin, l)
        self.dev = qml.device(device, wires=self.n_wires, shots=shots)

    def _dicke(self) -> None:
        qml.StatePrep(dicke_mixture(self.lin.m, self.w), wires=self.flags)

    def _load(self) -> None:
        """Flag i controls error_i <- sum_e c_i(e)|e>, then the flag is uncomputed."""
        for i in range(self.lin.m):
            qml.ctrl(qml.QubitUnitary, self.flags[i])(_local_prep(row_coeffs(self.lin, i), self.q), wires=self.err[i])
            qml.X(self.flags[i])
            qml.ctrl(qml.X, self.err[i], control_values=[0] * self.q)(self.flags[i])

    def _syndrome(self) -> None:
        p, q = self.lin.p, self.q
        for i, j in zip(*np.nonzero(self.lin.B)):
            c = int(self.lin.B[i, j])
            qml.QubitUnitary(_perm(p, q, lambda e, s, c=c: (s + c * e) % p), wires=self.err[i] + self.syn[j])

    def _decode(self) -> None:
        p, q = self.lin.p, self.q
        syn_wires = sum(self.syn, [])
        for s, e in self.table:
            ctrl_vals = sum((_bits(int(v), q) for v in s), [])
            for i in np.flatnonzero(e):
                qml.ctrl(qml.QubitUnitary, syn_wires, control_values=ctrl_vals)(_shift(p, q, -int(e[i])), wires=self.err[i])

    def _qft(self) -> None:
        for j in range(self.lin.n):
            qml.QubitUnitary(qft_p(self.lin.p, self.q), wires=self.syn[j])

    def ops(self, upto: str = "qft") -> None:
        for stage in STAGES[: STAGES.index(upto) + 1]:
            getattr(self, f"_{stage}")()

    def register_probs(self, register: str, upto: str = "qft") -> np.ndarray:
        """Distribution of one register after stage `upto`.

        register: "flags" -> shape (2,)*m; "error" -> (p,)*m; "syndrome" -> (p,)*n (one axis per F_p symbol).
        """
        cells = {"flags": [[w] for w in self.flags], "error": self.err, "syndrome": self.syn}[register]

        @qml.qnode(self.dev)
        def f():
            self.ops(upto)
            return qml.probs(wires=sum(cells, []))

        size = len(cells[0])
        pr = np.asarray(f()).reshape([2**size] * len(cells))
        return pr if register == "flags" else pr[np.ix_(*[range(self.lin.p)] * len(cells))]

    def probs_x(self) -> np.ndarray:
        """Output distribution over x in F_p^n (row index = base-p integer of x)."""
        return self.register_probs("syndrome").reshape(self.lin.p**self.lin.n)
