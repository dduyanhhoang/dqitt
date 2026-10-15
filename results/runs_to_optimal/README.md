# Qubits and runs-to-optimal on ttqaoa ensembles (DQITT-005)

Sources: `experiments/runs_to_optimal.py`, `experiments/qaoa_rerun.py`, `experiments/qaoa_energy_check.py`,
`experiments/summarize_runs.py`; data `$DQITT_DATA/ttqaoa/ensembles/inst_01..20`; run 2026-10-16.
Per instance and method: `ensembles.csv`. Medians: `summary.md`.

## Metric

A run (shot) is optimal when it decodes to a schedule with V = V* broken constraints (V* from
`classical.optimum`, see `results/classical_reference`). For each method we compute the exact output
distribution, then

- `p_opt`: probability that one run is optimal;
- `n50`, `n95`: smallest N with 1 − (1 − p_opt)^N ≥ 0.5 or 0.95 (runs needed to see an optimum at least once);
- `first_hit`, `hits`: in one sequence of 20,000 runs (QAOA re-run: its real measured shots; others: sampled
  from the exact distribution with a fixed seed);
- `p_valid`: probability that a run is a schedule at all (no section with two teachers, values in {0,1}).

QAOA training cost (restarts × COBYLA evaluations) is not counted in the runs.

## Methods

| method | what | qubits |
|---|---|---|
| random-eligible | uniform over eligible teachers per section | 0 |
| random-qubo | uniform over the 2^16 QUBO bitstrings | 0 |
| random-teacher-index | uniform over F_5^8 | 0 |
| qaoa-published-p{1..5} | ttqaoa circuit at ttqaoa's published angles, inst_01..10 | 16 |
| qaoa-rerun-p1 | ttqaoa `solve_qaoa` unchanged (20 restarts, 20,000 shots), p=1 | 16 |
| dqi-teacher-index-l1 | DQI on the no-quota formulation, l=1; equals the circuit (eps_1 = 0) | 104-140 |
| dqi-teacher-index-l2 | same at l=2 with an ideal decoder; the lookup table fails on eps_2 = 0.39-0.67 | 104-140 |
| dqi-binary-hard-l1 | DQI on the merged binary hard-quota formulation, l=1 (closed form) | 200-284 |

## Results

See `summary.md`. Ranges of p_opt over instances:

| method | p_opt min | p_opt max | median first hit (runs) | runs with no hit in 20,000 |
|---|---|---|---|---|
| random-eligible | 0 | 0.406 | 22.5 | 2 instances |
| random-qubo | 6.1e-5 | 0.0113 | 673 | 1 |
| qaoa-rerun-p1 | 1.0e-3 | 0.0811 | 59 | 0 |
| qaoa-published-p5 | 1.0e-5 | 0.358 | 20 | 1 |
| dqi-teacher-index-l1 | 9.1e-5 | 0.0076 | 724 | 0 |
| dqi-teacher-index-l2 | 5.0e-4 | 0.0215 | 253 | 0 |
| dqi-binary-hard-l1 | 1.2e-14 | 1.4e-7 | - | 20 |

- random-eligible has p_opt = 0 on 2 instances because every optimum there leaves a section empty or
  uses an ineligible teacher.
- dqi-binary-hard-l1: the state puts almost all weight outside {0,1}^16 (median p_valid 1.1e-8), so it
  essentially never returns a schedule.
- `../qaoa_rerun/energy_check.csv`: on inst_01..10 the published p=1 angles give <H> between 93 and 179,
  close to or above the uniform-state average (123 to 175); the re-optimised angles give 75 to 93.
  The published angles are not optimal for the current ttqaoa Hamiltonian; why is not established.
