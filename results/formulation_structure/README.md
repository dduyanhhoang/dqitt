# Formulation structure on ttqaoa ensembles (DQITT-003)

Source: `experiments/formulation_structure.py`, data `$DQITT_DATA/ttqaoa/ensembles/inst_01..20`, run 2026-10-16.
Raw per-instance values: `ensembles.csv`. Ranges over the 20 instances:

| formulation | constraints covered | n | m | p | d | instances with d=2 | ε₁ | ε₂ | DQI circuit qubits |
|---|---|---|---|---|---|---|---|---|---|
| teacher-index (`build`) | eligibility, conflict | 8 | 20–29 | 5 | 3 | 0 | 0 | 0.39–0.67 | 104–140 |
| binary, no quota | eligibility, conflict | 16 | 34–44 | 3 | 3 | 0 | 0 | 0.19–0.28 | 134–164 |
| binary, hard quota | all | 16 | 38–48 | 5–11 | 2–3 | 11 | 0–0.087 | 0.11–0.27 | 200–294 |
| binary, soft quota | all (quota as score) | 16 | 54–64 | 5–11 | 2 | 20 | 0.31–0.38 | 0.60–0.67 | 264–374 |
| binary, hard quota, merged rows | all | 16 | 38–48 | 5–11 | 3 | 0 | 0 | 0.11–0.14 | 200–284 |
| ttqaoa QUBO (QAOA) | all (penalties) | 16 | – | – | – | – | – | – | 16 (46–64 Pauli terms) |

- ε_k: fraction of weight-k errors an ideal minimum-weight decoder cannot recover uniquely.
- DQI circuit qubits = m + ⌈log₂p⌉(m+n), the register layout of `circuit.DQICircuit`.
- d=2 in the hard variant: a teacher's quota row has the same coefficients as a domain row
  (teacher with one eligible section) or a conflict row (two eligible sections that conflict).
