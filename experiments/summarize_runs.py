"""Aggregate results/runs_to_optimal/ensembles.csv per method into summary.md.

Medians over instances, on two subsets so every method in a table covers the same instances:
inst_01..10 (where ttqaoa published QAOA angles) and all 20.

    uv run python experiments/summarize_runs.py
"""

from __future__ import annotations

import csv
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

DIR = Path("results/runs_to_optimal")
FIRST10 = {f"inst_{i:02d}" for i in range(1, 11)}


def fmt(x: float) -> str:
    if math.isinf(x):
        return "∞"
    return f"{x:.3g}" if x < 1e4 else f"{x:.2e}"


def table(rows: list[dict], instances: set[str]) -> list[str]:
    by = defaultdict(list)
    for r in rows:
        if r["instance"] in instances:
            by[r["method"]].append(r)
    out = [
        "| method | qubits | instances | median P(opt) | median runs for 95% | median E[V] (valid) | P(opt) = 0 on |",
        "|---|---|---|---|---|---|---|",
    ]
    for method, rs in by.items():
        if len(rs) != len(instances):
            continue
        q = sorted({int(r["qubits"]) for r in rs})
        p = [float(r["p_opt"]) for r in rs]
        n95 = [float(r["n95"]) for r in rs]
        v = [float(r["mean_v_valid"]) for r in rs if r["mean_v_valid"] != "nan"]
        out.append(
            f"| {method} | {q[0]}{'-' + str(q[-1]) if len(q) > 1 else ''} | {len(rs)} | {fmt(st.median(p))} "
            f"| {fmt(st.median(n95))} | {fmt(st.median(v)) if v else '-'} | {sum(x == 0 for x in p)} |"
        )
    return out


def main() -> None:
    rows = list(csv.DictReader((DIR / "ensembles.csv").open()))
    every = {r["instance"] for r in rows}
    lines = ["## inst_01..10 (all methods)", "", *table(rows, FIRST10), "", "## all 20 instances", "", *table(rows, every)]
    (DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
