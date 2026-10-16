# dqitt — Decoded Quantum Interferometry for teacher–section timetabling

Same problem and data format as [QMOS-Lab/ttqaoa](https://github.com/QMOS-Lab/ttqaoa)
(teacher ↔ section assignment with time-slot conflicts, skill/availability eligibility and
teacher quota), solved with DQI (Jordan et al., arXiv:2408.08292) and compared with ttqaoa's QAOA.

## Formulations

- **Teacher-index** (`problem.build`): `y_j ∈ F_p` = teacher of section `j`, `p` = smallest prime ≥ #teachers
  (extra values are dummy teachers). Conflict row `y_a − y_b ∈ F_p \ {0}`, eligibility row `y_j ∈ E_j`.
  No quota: it is not a linear row in this encoding.
- **Binary** (`formulations.binary`): `x_{s,l} ∈ {0,1}` per eligible pair, rows for domain, one-hot,
  per-teacher conflict and quota (`hard`: `Σ_s x_{s,l} ∈ [Qmin, Qmax]`; `soft`: one row per load value).
  `merge=True` joins rows with equal coefficients, which otherwise give dual distance 2.

The original problem's objective, used to score every method, is `problem.violations`
(broken constraints of an assignment); `classical.optimum` / `classical.cpsat_optimum` give the optimum.

## Layout

```
src/dqitt/
  paths.py         data root ($DQITT_DATA, default /home/al/Data)
  problem.py       CSV loading, teacher-index LINSAT, violations, brute force, code distance, ε_k
  formulations.py  binary x_{s,l} encodings with quota
  dqi.py           ideal DQI state (enumeration), closed-form l=1 state for large p^n
  circuit.py       PennyLane F_p circuit (m + ⌈log₂p⌉(m+n) qubits, lookup-table decoder),
                   run stage by stage: dicke, load, syndrome, decode, qft
  classical.py     exact optimum, CP-SAT model, replica of ttqaoa's notebook CP-SAT model
  ttqaoa.py        adapter to ttqaoa's QUBO/QAOA (dependency group `baselines`)
  metrics.py       shots needed to see an optimal schedule
notebooks/         dqi_end_to_end.ipynb: data -> formulation -> each DQI stage -> results
experiments/       one script per study, each writes to results/<study>/
results/           CSV outputs + README per study
notes/             brief.md (theory), literature.md (cited facts)
```

## Data

Not in the repo. `$DQITT_DATA` (default `/home/al/Data`) holds:

- `dqitt/{tiny,mini,note001}`: hand-made instances (no quota).
- `dqitt/fpt_toy/{day,am}`: an FPT-style Monday built from the ttqaoa toy set by the notebook (part 2).
- `ttqaoa/`: ttqaoa `data/raw` and `data/processed` at commit `3b494c1` (see `SOURCE.md`).

## Run

```bash
uv sync --all-groups
uv run pytest
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/dqi_end_to_end.ipynb   # ~2 min
uv run --group baselines python experiments/formulation_structure.py
uv run --group baselines python experiments/classical_reference.py
OMP_NUM_THREADS=1 uv run --group baselines python experiments/qaoa_rerun.py   # hours
uv run --group baselines python experiments/runs_to_optimal.py
```
