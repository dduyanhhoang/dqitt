# DQI literature evidence (read 2026-10-16)

Facts only, each with its source. Basis for the Notion survey (LIT-003).

## Jordan et al., arXiv:2408.08292v5
- Def. 2.1 max-LINSAT: B ∈ F_p^{m×n}, F_i ⊂ F_p, maximise #{i : b_i·x ∈ F_i}. max-XORSAT = p=2, |F_i|=1. [§2 p.4]
- Thm 4.1 semicircle law: |F_i| = r for all i, polynomial degree ℓ, **2ℓ+1 < d^⊥** (d^⊥ = min distance of C^⊥ = {d : Bᵀd = 0}); as m→∞ with ℓ/m fixed, ⟨s⟩/m = (√((ℓ/m)(1−r/p)) + √((r/p)(1−ℓ/m)))² if r/p ≤ 1−ℓ/m, else 1. [§4 eqs 9–10 p.11]
- Lemma 9.2 finite m (same condition): ⟨s⟩ = mr/p + (√(r(p−r))/p)·w†A^{(m,ℓ,d)}w, A tridiagonal, diag kd, off-diag √(k(m−k+1)), d = (p−2r)/√(r(p−r)); optimal w = principal eigenvector. [§9.2 eqs 69–70 p.27]
- State norm = ‖w‖ when 2ℓ < d^⊥. [Lemma 9.1 p.26]
- Equal |F_i| = r assumed "to keep the presentation as simple as possible". [§8.2.1 p.21]
- Steps: weight register Σ w_k|k⟩ → Dicke D(m,k) → uncompute k → G_i: |1⟩ ↦ Σ_y g̃_i(y)|y⟩ → syndrome Bᵀy → reversible decode → (inverse) QFT on syndrome. Error register m⌈log₂p⌉ qubits, syndrome n⌈log₂p⌉; mask = LSB of each error subregister. Gates O(T + m²), T = decoder cost. [§8.2.2 p.23; §1 p.4; §9.1 eq 65]
- p=2 decoding failure: postselect error register on 0; ε_k = |F_k|/C(m,k); success prob Σ w_k²(1−ε_k). Thm 10.1 (p=2, random v): E⟨f⟩ ≥ w†(A^{(m,ℓ,0)} − 2ε(m+1))w / Σ w_k²(1−ε_k). [§10.3 Lemma 10.4 Thm 10.1 p.39–40]
- OPI: Vandermonde B, C^⊥ Reed–Solomon, ℓ = ⌊(n+1)/2⌋; at n/p≈1/10, r/p≈1/2 DQI 0.7179 vs Prange 0.55; p=521 ≈ 10⁸ Toffoli, 9×10³ logical qubits. [§2 p.5–6, §5, Fig. 3, §16]
- max-XORSAT 31,216 vars / 50,000 constraints: DQI+BP ≥ 0.831; SA 0.764 (8 s), matched only after 73 h on 5 cores; tailored heuristic 0.880 in 7 min; Prange 0.812. Not claimed as advantage. [Table 1 p.8; §12]
- k-local CSP → weighted max-k-XORSAT by Hadamard transform [p.6–7]; theorems only for unweighted, equal |F_i|.
- Output is unbiased: equal-objective solutions are equally likely. [p.10]

## PennyLane tutorial (pennylane.ai/qml/demos/tutorial_dqi, D. Angulo, 2025-09-19, upd. 2026-04-16)
- max-XORSAT, B rows 1000/1100/0110/0011/0001, v = 10101, m=5, n=4, ℓ=2; best 4/5.
- w = principal eigenvector of A (p=2, r=1, d=0) = [0.4385, 0.7071, 0.5547].
- 11 qubits: weight 0–1, error 2–6, syndrome 7–10. StatePrep → Dicke via SCS blocks under ctrl → uncompute weight → Z phase if v_i → CNOT Bᵀy → LUT decoder (controlled-X) → Hadamard → 100-shot counts vs 100 uniform samples. No expectation value reported.

## DQI-Kit, arXiv:2605.16955 (Thelen, Mauerer)
- Encodes to (weighted) max-LINSAT over F_q (Sage); estimates ⟨s⟩ from Jordan's closed form, no circuits. Decoders BP, ISD, LUT, nearest-neighbour; baselines brute force, OR-Tools CP, SA, Prange. [§VI]
- Exact when ℓ < d_min/2, else sampling of error vectors. [p.10]
- Standard DQI handles neither weights nor unequal |F_i|; duplicating constraints gives min distance 2; gadget (Thm 1) adds k variables + k constraints to raise distance by k. [§IV–V p.7]

## Limits and negative results
- Anschuetz, Gamarnik, Lu, arXiv:2509.14509: random max-k-XORSAT, OGP obstructs DQI above μ_top (Thm 62); AMP beats DQI numerically; depth-1 QAOA provably beats DQI at large k.
- Parekh, arXiv:2509.19966: MaxCut DQI needs girth ≥ 2ℓ+1 (Lemma 2); (1/2+ε)m needs girth linear in m (Thm 2); such graphs are classically exactly solvable (Thm 3).
- Kramer, Schubert, Eisert, arXiv:2606.13570: beyond r/q + O(1/√D) on bounded-degree max-LINSAT is NP-hard (Thm 8); classical-decoder DQI ceiling r/q + O(1/√(D log D)); dual distance O(1) reduces DQI to random. [§4.3, §5]

## Circuits and resources
- Patamawisut et al., arXiv:2504.18334: F₂ circuits, MaxCut ℓ=2, Gauss–Jordan decoder ≤ nm CNOT; ≤ 30 qubits simulated; at m=15 GJ 46 gates vs LUT 67,027; ≈ 2m qubits, O(m²) gates; postselects error = 0. Sabater et al. (arXiv:2509.08328 App. 10.1): GJ always succeeds only for full-rank square B.
- Khattar et al., arXiv:2510.10967: OPI over GF(2^b), m=4095, n=70, b=12 → 5.72×10⁶ Toffoli, 1,885 logical qubits; sparse Dicke (ℓ index registers) avoids the m·b error register. [§II.A, Table VI]
- Sabater et al., arXiv:2509.08328: BP1 qubits N_q = (T+1)m + Tn + 2⌈log₂(t+1)⌉ (eq 71); automotive option-package ILP → max-XORSAT, all instances d=3; Gurobi far better. [§8 p.40]
- Gao, Chang et al., arXiv:2604.15025: F₂ bound R ≥ 1/2 + (λ/2m)(1−2ε̄)/(1−ε̄), ε̄ = Σ w_k² ε_k. [Thm 1]
- Bu, Gu, Li, arXiv:2605.10666: multivariate DQI for weighted max-LINSAT; still 2ℓ+1 < d^⊥ and equal r. [Thm 2]
