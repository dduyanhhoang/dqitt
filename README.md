# dqitt — Decoded Quantum Interferometry for teacher–section timetabling

Same problem and data format as [QMOS-Lab/ttqaoa](https://github.com/QMOS-Lab/ttqaoa)
(teacher ↔ section assignment with time-slot conflicts and skill/availability
eligibility), solved with DQI (Jordan et al., arXiv:2408.08292) instead of QAOA.
The teacher quota constraint is **excluded**: it is not expressible as max-LINSAT.

## Formulation

- `y_j ∈ F_p` = teacher of section `j`, `p` = smallest prime ≥ #teachers (extra values are dummy teachers).
- Conflict row for each pair of sections sharing a slot: `y_a − y_b ∈ F_p \ {0}`.
- Eligibility row for each section: `y_j ∈ E_j` (skill ≥ 5 and available; disable with `--no-eligibility`).
- Objective: maximise the number of satisfied rows.

## Structure

```mermaid
flowchart LR
    D[(data/*/ *.csv<br/>Task, SlotConflict,<br/>InstructorSkill, InstructorSlot)] --> P[problem.py<br/>Instance → Linsat B, F_i over F_p<br/>brute force, code distance, ε_k]
    P --> Q[dqi.py<br/>tridiagonal A^(m,ℓ,d) weights<br/>exact ideal state |P(f)⟩ in numpy]
    P --> C[circuit.py<br/>PennyLane F_p circuit<br/>Dicke → local prep → Bᵀ → LUT decode → QFT_p]
    Q --> V[viz.py<br/>histogram, conflict graph, circuit drawing]
    C --> V
    Q --> CLI[cli.py  `dqitt DATA --l L [--circuit]`]
    C --> CLI
    CLI --> R[(reports/*.png)]
```

Two evaluation paths:

1. **Ideal DQI** (`dqi.py`): builds `|P(f)⟩ = Σ_k w_k |P_k⟩` directly in the `x` basis for
   `p^n ≲ 10^7`. Decoder-independent. `w` is chosen by an exact generalized eigenproblem, which
   is also correct when the `|P_k⟩` are not orthogonal (short cycles in the conflict graph).
   The paper's tridiagonal formula is printed alongside; it is only valid when `2ℓ+1 < d_code`.
2. **Circuit** (`circuit.py`): the full DQI circuit with `⌈log₂p⌉` qubits per `F_p` symbol and a
   lookup-table decoder. `m + ⌈log₂p⌉(m+n)` qubits, so tiny instances only (`data/tiny`: 24 qubits).
   Its output distribution equals the ideal state whenever the decoder succeeds (tested at ℓ=1).

## Run

```bash
uv sync
uv run pytest
uv run dqitt data/mini --l 2                       # ideal DQI, 8 sections / 3 teachers
uv run dqitt data/tiny --l 1 --circuit             # + PennyLane circuit, 3 sections
uv run dqitt data/tiny --l 1 --circuit --device lightning.gpu   # HPC (needs pennylane-lightning-gpu)
```

Outputs in `reports/`: `histogram.png` (satisfied-constraint distribution: random vs DQI vs circuit),
`conflict_graph.png` (coloured by best sampled assignment), `circuit.png`.

## Datasets

| dir | sections | teachers | p | rows m | note |
|---|---|---|---|---|---|
| `data/tiny` | 3 | 3 | 3 | 6 | triangle; circuit-simulable |
| `data/mini` | 8 | 3 | 3 | 20 | two K4 slot cliques, optimum 18/20 |

Both use the ttqaoa CSV layout minus `InstructorQuota.csv`.
