"""DQI target state and optimal weights (Jordan et al., arXiv:2408.08292, Sec. 8.1).

Per row i let h_i(x) = [b_i.x in F_i] and g~_i = (h_i - mean)/std, so that
    s(x) = #satisfied = sum_i h_i(x) = m r/p + sigma * sum_i g~_i(x).
The DQI state is |psi> = sum_k w_k |P_k>,  |P_k> = p^{-n/2} C(m,k)^{-1/2} sum_x e_k(g~(x)) |x>,
with e_k the elementary symmetric polynomial. If all row images B^T e (|e|<=l) are
distinct, the |P_k> are orthonormal and sum_i g~_i acts as the tridiagonal A^{(m,l,d)}.
"""

from __future__ import annotations

from itertools import combinations, product
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


def _row_stats(lin: Linsat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """g~_i(v) for every row and value v in F_p, shape (m, p); and per-row mean, std of h_i."""
    mean = lin.F.mean(axis=1)
    std = np.sqrt(mean * (1 - mean))
    return (lin.F - mean[:, None]) / std[:, None], mean, std


def l1_weights(lin: Linsat) -> np.ndarray:
    """Optimal (w_0, w_1) at l=1 without enumerating F_p^n.

    Needs no two proportional rows (d >= 3): then |P_0>, |P_1> are orthonormal (Jordan et al.
    Lemma 9.1, 2l < d) and <s> = w^T M w with
        M00 = sum mu_i,  M01 = sum sigma_i / sqrt(m),  M11 = sum mu_i + (1/m) sum_l sigma_l T_l,
    T_l = sum_{i,j} E[g~_l g~_i g~_j]: the i = j = l term plus ordered pairs i != j with
    b_l = a b_i + c b_j (any other triple has independent factors and averages to 0).
    """
    g, mu, sigma = _row_stats(lin)
    p, m = lin.p, lin.m
    B = lin.B % p
    T = (g**3).mean(axis=1)
    by_row: dict[bytes, list[int]] = {}
    for l, b in enumerate(B):
        by_row.setdefault(b.tobytes(), []).append(l)
    u = np.arange(p)
    for i, j in combinations(range(m), 2):
        for a, c in product(range(1, p), repeat=2):
            for l in by_row.get(((a * B[i] + c * B[j]) % p).tobytes(), ()):
                lin_comb = (a * u[:, None] + c * u[None, :]) % p
                T[l] += 2 * (g[i][:, None] * g[j][None, :] * g[l][lin_comb]).mean()
    off = sigma.sum() / sqrt(m)
    M = np.array([[mu.sum(), off], [off, mu.sum() + (sigma * T).sum() / m]])
    w = np.linalg.eigh(M)[1][:, -1]
    return w * np.sign(w[0] or 1)


def l1_probs(lin: Linsat, w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Probability of each x in X (..., n) under the l=1 DQI state, valid under l1_weights' condition."""
    g, _, _ = _row_stats(lin)
    G = g[np.arange(lin.m), (X @ lin.B.T) % lin.p].sum(axis=-1)
    return (w[0] + w[1] * G / sqrt(lin.m)) ** 2 / lin.p**lin.n
