# DQI on teacher–section assignment: core points

## 1. Formulation (max-LINSAT over 𝔽ₚ)

- $y_j \in \mathbb{F}_p$: index of the teacher assigned to section $j$; $p$ prime, $p \ge$ number of teachers (pad with dummy teachers if needed).
- Each pair of sections $(a,b)$ sharing a time slot gives one row of $B$: $y_a - y_b \in F_i = \mathbb{F}_p\setminus\{0\}$. Every row has exactly two nonzero entries ($+1,-1$); $r = |F_i| = p-1$.
- Objective: maximize the number of satisfied rows. No one-hot penalty is needed. Quota constraints (max sections per teacher) cannot be expressed in LINSAT (Prop. 1) → excluded from the DQI part; state this explicitly in the report.
- Since each row involves only two variables, $B$ is the (signed) incidence matrix of the conflict graph $G$ (vertices = sections, edges = conflicting pairs).

## 2. DQI

- Target state: $|P(f)\rangle = \sum_x P(f(x))|x\rangle$, $P$ of degree $\ell$. Larger $\ell$ concentrates more probability on good solutions.
- Prepared via the Fourier side: $\sum_{k\le\ell} \frac{w_k}{\sqrt{\binom{m}{k}(p-1)^k}}\sum_{|e|=k} (\ldots)|B^T e\rangle$. Requires **decoding**: recover $e$ from $s = B^T e$ to uncompute the error register. This is syndrome decoding with parity-check matrix $B^T$.
- Optimal $w_k$ = principal eigenvector of the tridiagonal matrix $A^{(m,\ell,d)}$ with $d = \frac{p-2r}{\sqrt{r(p-r)}}$ (Jordan et al. Sec. 8.1); its largest eigenvalue gives the expected satisfied fraction. Random baseline $= r/p = 1-1/p$.
- Unique decoding requires $2\ell < d_{\text{code}}$, where $d_{\text{code}}$ is the minimum distance of $\ker(B^T)$.

## 3. The negative result

- $\ker(B^T)$ is the *cycle code* of $G$ → $d_{\text{code}} = $ girth$(G)$.
- Several sections in one time slot ⇒ triangles ⇒ girth 3 ⇒ $\ell = 1$ under unique decoding.
- With $\ell=1$ the satisfied fraction tends to $1-1/p$ as $m$ grows (m=1000, p=3: ≈69.6% vs 66.7%). This is a structural limit, not an artifact of one instance.

## 4. Tightening the argument: $\ell > 1$ with an imperfect decoder

- Thm 7.1 (Jordan et al.) and arXiv:2604.15025: DQI still yields good solutions if the decoder fails only with small rate $\varepsilon_k$ on weight-$k$ layers.
- For a cycle code, two errors $e \ne e'$ collide iff $e - e'$ is a combination of cycles. For $k \ll m$ with scattered errors, collisions are rare → in principle $\ell > 1$ is usable.
- But beating random substantially needs $\ell \sim$ const $\times m$; errors are then dense and collisions through triangles are frequent. Measure $\varepsilon_k$ numerically rather than stopping at "d=3".

## 5. Decoder: why not a LUT, and what replaces it

- The PennyLane demo's lookup table has $p^m$ entries, one multi-controlled gate each → feasible only for $m \lesssim 15$.
- Original paper: sparse $B$ → LDPC code → **belief propagation**; OPI → Reed–Solomon decoders (not applicable here, the code is not RS).
- For this problem: BP over 𝔽ₚ with Tanner graph checks = vertices of $G$, variables = edges (degree exactly 2). This is the weakest configuration for BP (degree-2 variables + short cycles) → poor results are a result, not a bug. Second baseline: information-set decoding (Prange).
- Measure $\varepsilon_k$ vs $k$, plug into the Thm 7.1 lower bound → main figure of the presentation.

## 6. Verification

- Small instances ($n\le 8$, $p=3$): build $|P(f)\rangle$ directly with numpy (ideal DQI, decoder-independent), compare with random and with the formula in §2.
- 𝔽ₚ circuit only for the 3-section example if time permits (2 qubits/symbol, QFT over $\mathbb{Z}_p$, controlled addition mod $p$).
- Put side by side on the same dataset: QAOA (existing), predicted DQI at feasible $\ell$, random.

## References

- Jordan et al., arXiv:2408.08292 / Nature 2025 — Sec. 8.1, Thm 7.1, 𝔽_q generalization.
- PennyLane `tutorial_dqi` — 𝔽₂ circuit, LUT decoder.
- Patamawisut et al., arXiv:2504.18334 — explicit reversible circuits.
- arXiv:2604.15025 — $\ell$ beyond the decoding threshold, $\varepsilon_k$.
- arXiv:2607.28120 — approximate DQI sampling via MCMC (large-scale evaluation).
- DQI-Kit, arXiv:2605.16955 — constraint → Max-LINSAT encoding and performance estimates for cross-checking.

## 7. Findings from the first implementation (2026-09-11)

- Eligibility (skill + availability) *is* expressible: one unary row $y_j \in E_j$ per section. Only quota is dropped. Dummy teachers from padding $p$ are excluded by these rows automatically.
- Jordan et al. v3 Lemma 9.2: the tridiagonal formula $\langle s\rangle/m = r/p + \tfrac{\sqrt{r(p-r)}}{p}\lambda_{\max}/m$ is exact only for $2\ell+1 < d_{\text{code}}$. With girth 3 it already **overestimates at $\ell=1$** (tiny: 0.833 predicted vs 0.811 exact; mini conflict-only: 0.790 vs 0.750), because $\langle P_1|\sum_i \tilde g_i|P_1\rangle$ picks up $E[\tilde g_a\tilde g_b\tilde g_c]\neq 0$ on every triangle. Use the exact generalized eigenproblem ($M w = \lambda G w$, $M_{kk'}=\langle P_k|s|P_{k'}\rangle$, $G$ = Gram) implemented in `dqi.py`; it also gives the optimal $w$ for heterogeneous $|F_i|$.
- Ideal min-weight decoder failure on mini ($\ell=2$): $\varepsilon_2 \approx 0.47$; on tiny $\varepsilon_2 = 1$. Circuit output matches the ideal state exactly at $\ell=1$ and degrades to $\langle s\rangle/m = 0.78$ (vs ideal 0.92) at $\ell=2$ — the negative result of §3 reproduced in a real simulation.
- Theory references verified: Thm 7.1 in the published SM = Thm 10.1 in v3 (p=2 only, bound $\langle s\rangle/m \ge 1/2+\sqrt{\mu(1-\mu)}-\varepsilon$). arXiv:2604.15025 (Gao & Chang) gives the tighter master bound $E\langle f\rangle \ge [\lambda_{\max}(1-2\bar\varepsilon)+2d\bar\eta]/(1-\bar\varepsilon)$ with $\bar\varepsilon=\sum_k \varepsilon_k w_k^2$ — but its $A$-matrix carries a $(q-1)$ factor absent in Jordan's; don't mix conventions. arXiv:2504.18334 has $\mathbb{F}_2$ circuits only (Gauss–Jordan reversible decoder); no $\mathbb{F}_p$ circuits exist in the literature, ours in `circuit.py` is built from small `QubitUnitary` blocks (QFT$_p$, controlled mod-$p$ addition, Householder local prep).
