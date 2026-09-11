"""DQI target state and optimal weights (Jordan et al., arXiv:2408.08292, Sec. 8.1).

Per row i let h_i(x) = [b_i.x in F_i] and g~_i = (h_i - mean)/std, so that
    s(x) = #satisfied = sum_i h_i(x) = m r/p + sigma * sum_i g~_i(x).
The DQI state is |psi> = sum_k w_k |P_k>,  |P_k> = p^{-n/2} C(m,k)^{-1/2} sum_x e_k(g~(x)) |x>,
with e_k the elementary symmetric polynomial. If all row images B^T e (|e|<=l) are
distinct, the |P_k> are orthonormal and sum_i g~_i acts as the tridiagonal A^{(m,l,d)}.
"""

from __future__ import annotations

from math import comb, sqrt

import numpy as np
from scipy.linalg import eigh

from .problem import Linsat, all_x


def tridiagonal(m: int, l: int, d: float) -> np.ndarray:
    k = np.arange(l + 1)
    A = np.diag(k * d).astype(float)
    off = np.sqrt(k[1:] * (m - k[1:] + 1))
    A += np.diag(off, 1) + np.diag(off, -1)
    return A


def theory_weights(m: int, l: int, p: int, r: int) -> tuple[np.ndarray, float]:
    """Optimal w_k and predicted expected satisfied fraction <s>/m for homogeneous |F_i| = r."""
    d = (p - 2 * r) / sqrt(r * (p - r))
    vals, vecs = np.linalg.eigh(tridiagonal(m, l, d))
    w = vecs[:, -1] * np.sign(vecs[0, -1] or 1)
    frac = r / p + sqrt(r * (p - r)) / p * vals[-1] / m
    return w, frac


def normalized_rows(lin: Linsat, X: np.ndarray) -> np.ndarray:
    """g~_i(x) for all x, shape (p^n, m)."""
    h = lin.satisfied(X).astype(float)
    mean = lin.F.mean(axis=1)
    std = np.sqrt(mean * (1 - mean))
    return (h - mean) / std


def basis_states(lin: Linsat, l: int, X: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Columns |P_0>..|P_l> in the x basis, shape (p^n, l+1)."""
    X = all_x(lin.p, lin.n) if X is None else X
    g = normalized_rows(lin, X)
    e = np.zeros((len(X), l + 1))
    e[:, 0] = 1
    for i in range(lin.m):  # e_k <- e_k + g_i e_{k-1}
        e[:, 1:] += g[:, i : i + 1] * e[:, :-1]
    norm = np.array([sqrt(comb(lin.m, k)) for k in range(l + 1)]) * lin.p ** (lin.n / 2)
    return e / norm, X


def ideal_state(lin: Linsat, l: int, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact normalized DQI state in the x basis.

    If w is None, w maximizes <s> over the span of |P_k> (generalized eigenproblem,
    exact even when the |P_k> are not orthogonal). Returns (psi, X, w).
    """
    P, X = basis_states(lin, l)
    s = lin.count(X).astype(float)
    if w is None:
        G = P.T @ P
        M = P.T @ (s[:, None] * P)
        vals, vecs = eigh(M, G)
        w = vecs[:, -1]
        w = w / np.linalg.norm(w) * np.sign(w[0] or 1)
    psi = P @ w
    return psi / np.linalg.norm(psi), X, w


def expected_satisfied(psi: np.ndarray, lin: Linsat, X: np.ndarray) -> float:
    return float((np.abs(psi) ** 2 * lin.count(X)).sum())


def sample(psi: np.ndarray, X: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(len(psi), size=shots, p=np.abs(psi) ** 2)
    return X[idx]
