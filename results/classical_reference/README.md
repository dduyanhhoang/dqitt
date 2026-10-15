# Classical references on ttqaoa ensembles (DQITT-004)

Source: `experiments/classical_reference.py`, data `$DQITT_DATA/ttqaoa/ensembles/inst_01..20`, run 2026-10-16.
Per-instance values: `ensembles.csv`.

Objective: `problem.violations`, the number of broken constraints (section without an eligible
teacher, conflicting pair on one teacher, teacher outside quota) over all assignments in {-1, 0..3}^8.

- Exact optimum V* by enumeration (5^8 assignments): 0 on 4 instances (05, 06, 17, 19; 4 optimal
  schedules each), 1 on 8, 2 on 7, 3 on 1. Optimal schedules per instance: 4 to 742.
- `cpsat_optimum` (CP-SAT, same objective) agrees with V* on 20/20, 0.006 to 0.010 s each.
- `cpsat_ttqaoa_notebook` (ttqaoa notebook cell 18 plus the missing Minimize/Solve): OPTIMAL for its
  own objective on 20/20, but its schedules break 2 to 4 constraints and match V* on 0/20, including
  the 4 feasible instances. Its conflict term W²(x₁+x₂+2z) charges W² per conflicting pair a section
  belongs to even without a clash; with W_section = W_conflict = 300 this cancels the W² reward for
  assigning a section, so sections are left empty (inst_05: two sections unassigned, V = 2 vs V* = 0).
- ttqaoa QUBO ground state (exact diagonal of `TimetableProblem.cost_h`, W = 3, 2, 1): an optimal
  schedule on 10/20 instances. On the other 10 its ground states break 1 to 3 more constraints than
  V*. inst_04: an optimal schedule (V = 1) has energy 62, the ground state (V = 2) has 56.
