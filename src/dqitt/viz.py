from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .problem import Instance, Linsat


def satisfied_histogram(lin: Linsat, samples: dict[str, np.ndarray], optimum: int, path: Path) -> None:
    """samples: label -> (shots, n) array of x."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(-0.5, lin.m + 1.5)
    for label, xs in samples.items():
        ax.hist(lin.count(xs), bins=bins, alpha=0.5, density=True, label=label)
    ax.axvline(optimum, color="k", ls="--", label=f"optimum = {optimum}")
    ax.set_xlabel("satisfied constraints")
    ax.set_ylabel("probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def conflict_graph(inst: Instance, path: Path, assignment: np.ndarray | None = None) -> None:
    G = nx.Graph(inst.conflicts)
    G.add_nodes_from(range(len(inst.sections)))
    colors = assignment if assignment is not None else "lightgray"
    fig, ax = plt.subplots(figsize=(5, 4))
    nx.draw(G, nx.spring_layout(G, seed=0), ax=ax, with_labels=True, node_color=colors, cmap="Set2",
            labels={i: f"{s}\n{sl}" for i, (s, sl) in enumerate(zip(inst.sections, inst.slots))}, font_size=7)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def circuit(circ, path: Path) -> None:
    out = circ.draw(style="pennylane", max_length=80)
    figs = [out[0]] if isinstance(out, tuple) else [f for f, _ in out]  # max_length may split into chunks
    for i, fig in enumerate(figs):
        fig.savefig(path.with_stem(f"{path.stem}_{i}") if len(figs) > 1 else path, dpi=120, bbox_inches="tight")
        plt.close(fig)
