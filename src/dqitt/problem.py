"""Timetable instance -> max-LINSAT over F_p.

Variable y_j in F_p = teacher of section j. Rows:
  conflict (a, b):  y_a - y_b in F_p \\ {0}       (two nonzero entries)
  eligibility j:    y_j in E_j                     (one nonzero entry)
Quota is not expressible in LINSAT and is excluded.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import combinations, product
from math import comb
from pathlib import Path

import numpy as np
from loguru import logger


@dataclass
class Instance:
    sections: list[str]
    slots: list[str]
    teachers: list[str]
    eligible: list[set[int]]  # per section: eligible teacher indices
    conflicts: list[tuple[int, int]]  # section pairs sharing a time slot
    quota: list[tuple[int, int]] | None = None  # per teacher (min, max) sections


def _matrix(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    rows = list(csv.reader(path.open()))
    cols = rows[0][1:]
    idx = [r[0] for r in rows[1:]]
    m = np.array([[int(float(v or 0)) for v in r[1:]] for r in rows[1:]])
    return idx, cols, m


def _column(m: np.ndarray, cols: list[str], key: str) -> np.ndarray:
    """Column `key` of m, zeros when absent (ttqaoa treats a missing slot/subject as level 0)."""
    return m[:, cols.index(key)] if key in cols else np.zeros(len(m), int)


def _quota(path: Path, teachers: list[str]) -> list[tuple[int, int]] | None:
    if not path.exists():
        return None
    rows = {r[""].strip(): r for r in csv.DictReader(path.open())}
    return [(int(rows[t]["Min quota"]), int(rows[t]["Max quota"])) for t in teachers]


def load(data_dir: str | Path, threshold: int = 5) -> Instance:
    d = Path(data_dir)
    tasks = list(csv.DictReader(d.joinpath("Task.csv").open()))
    sections = [t["Class"].strip() for t in tasks]
    slots = [t["Slot"].strip() for t in tasks]
    subjects = [t["Subject"].strip() for t in tasks]
    teachers, skill_cols, skill = _matrix(d / "InstructorSkill.csv")
    _, slot_cols, avail = _matrix(d / "InstructorSlot.csv")
    conf_idx, conf_cols, conf = _matrix(d / "SlotConflict.csv")

    eligible = [
        set(np.flatnonzero((_column(skill, skill_cols, subj) >= threshold) & (_column(avail, slot_cols, slot) >= threshold)).tolist())
        for subj, slot in zip(subjects, slots)
    ]
    conflicts = [
        (a, b)
        for a, b in combinations(range(len(sections)), 2)
        if conf[conf_idx.index(slots[a]), conf_cols.index(slots[b])]
    ]
    logger.info(f"{d}: {len(sections)} sections, {len(teachers)} teachers, {len(conflicts)} conflicts")
    return Instance(sections, slots, teachers, eligible, conflicts, _quota(d / "InstructorQuota.csv", teachers))


def violations(inst: Instance, a: np.ndarray) -> int:
    """Broken constraints of the original problem for assignment a (teacher index per section, -1 = none).

    One per section without an eligible teacher, per conflicting pair on the same teacher,
    and per teacher whose load is outside its quota.
    """
    v = sum(t < 0 or t not in E for t, E in zip(a, inst.eligible))
    v += sum(a[s] == a[u] >= 0 for s, u in inst.conflicts)
    if inst.quota:
        load_ = np.bincount(a[a >= 0], minlength=len(inst.teachers))
        v += sum(not lo <= c <= hi for c, (lo, hi) in zip(load_, inst.quota))
    return int(v)


def next_prime(k: int) -> int:
    k = max(k, 2)
    while any(k % q == 0 for q in range(2, int(k**0.5) + 1)):
        k += 1
    return k


@dataclass
class Linsat:
    p: int
    B: np.ndarray  # (m, n) entries in F_p
    F: np.ndarray  # (m, p) bool: allowed values of b_i . x
    constant: int = 0  # dropped rows that are always satisfied

    @property
    def m(self) -> int:
        return self.B.shape[0]

    @property
    def n(self) -> int:
        return self.B.shape[1]

    def satisfied(self, x: np.ndarray) -> np.ndarray:
        """x: (..., n) -> (..., m) bool."""
        bx = (x @ self.B.T) % self.p
        return self.F[np.arange(self.m), bx]

    def count(self, x: np.ndarray) -> np.ndarray:
        return self.satisfied(x).sum(axis=-1)


def build(inst: Instance, eligibility: bool = True) -> Linsat:
    n, p = len(inst.sections), next_prime(len(inst.teachers))
    rows, allowed = [], []
    for a, b in inst.conflicts:
        r = np.zeros(n, int)
        r[a], r[b] = 1, p - 1
        rows.append(r)
        allowed.append(np.arange(p) != 0)
    constant = 0
    if eligibility:
        for j, E in enumerate(inst.eligible):
            if len(E) == p:
                constant += 1
                continue
            if not E:
                logger.warning(f"section {inst.sections[j]} has no eligible teacher; row dropped")
                continue
            r = np.zeros(n, int)
            r[j] = 1
            rows.append(r)
            allowed.append(np.isin(np.arange(p), list(E)))
    lin = Linsat(p, np.array(rows), np.array(allowed), constant)
    logger.info(f"LINSAT: p={p}, n={n}, m={lin.m} (+{constant} trivial rows)")
    return lin


def all_x(p: int, n: int) -> np.ndarray:
    """Every x in F_p^n, shape (p^n, n), row index = base-p integer of x."""
    return np.array(list(product(range(p), repeat=n)), dtype=int).reshape(p**n, n)


def brute_force(lin: Linsat) -> tuple[int, np.ndarray]:
    X = all_x(lin.p, lin.n)
    c = lin.count(X)
    return int(c.max()), X[c == c.max()]


def error_patterns(m: int, p: int, k: int) -> np.ndarray:
    """All e in F_p^m with Hamming weight k, shape (C(m,k)(p-1)^k, m)."""
    out = np.zeros((comb(m, k) * (p - 1) ** k, m), dtype=int)
    i = 0
    for supp in combinations(range(m), k):
        for vals in product(range(1, p), repeat=k):
            out[i, list(supp)] = vals
            i += 1
    return out


def decoder_failure(lin: Linsat, l: int) -> np.ndarray:
    """eps_k, k=0..l: fraction of weight-k errors an ideal min-weight decoder
    cannot recover uniquely from the syndrome B^T e (collisions with weight <= l)."""
    seen: dict[bytes, tuple[int, int]] = {}  # syndrome -> (min weight, count at min weight)
    layers = [error_patterns(lin.m, lin.p, k) for k in range(l + 1)]
    for k, E in enumerate(layers):
        for s in map(np.ndarray.tobytes, (E @ lin.B) % lin.p):
            w, c = seen.get(s, (k, 0))
            seen[s] = (k, c + 1) if w == k else (w, c)
    eps = np.zeros(l + 1)
    for k, E in enumerate(layers):
        S = (E @ lin.B) % lin.p
        fail = sum(seen[s.tobytes()] != (k, 1) for s in S)
        eps[k] = fail / len(E)
    return eps


def code_distance(lin: Linsat, max_w: int = 4) -> int | None:
    """Min weight of a nonzero e with B^T e = 0, searched up to max_w."""
    for k in range(1, max_w + 1):
        E = error_patterns(lin.m, lin.p, k)
        if (((E @ lin.B) % lin.p) == 0).all(axis=1).any():
            return k
    return None
