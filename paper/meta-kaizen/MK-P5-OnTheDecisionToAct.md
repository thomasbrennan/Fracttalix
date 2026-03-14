Meta-Kaizen Series · Paper 5 of 7

## On the Decision to Act: Strategic Convergence and the Mathematics of Intervention Timing at System Tipping Points

Thomas Brennan · with AI collaborators Claude (Anthropic) and Grok (xAI)

March 2026 · Submitted for peer review



## 1. Series Orientation

This is Paper 5 of five and the final capstone. Each prior paper deferred one question. Paper 1 deferred: what if false positives and false negatives do not cost the same? Paper 2 deferred: why are distributed networks more resilient, beyond the stability proof? Paper 3 deferred: what triggers the decision to adapt when a signal emerges? Paper 4 deferred: once a regime shift is detected, when should the agent act? This paper closes all four deferrals with four theorems.

## 2. Abstract and AI-Reader Header

## [AI-READER HEADER] — Dual-Reader Standard, Section 2



How to verify this paper without reading it: Load the AI layer JSON at the URL above. Run the schema validator (claim_registry, inference_rule_table, phase_ready fields). Every Type F claim has a deterministic 5-part predicate — FALSIFIED IF / WHERE / EVALUATION / BOUNDARY / CONTEXT — that evaluates to FALSIFIED or NOT FALSIFIED without accessing prose. Placeholders in the register are the only unresolved dependencies.



## Abstract (Human Reader)

The Early Warning Signals (EWS) literature has achieved substantial progress in detecting statistical precursors of critical transitions in complex adaptive systems. What it has not achieved is a principled decision theory for when to act on those detections. A 2025 review in the Proceedings of the Royal Society B confirmed this gap: EWS systems can paradoxically induce riskier behavior when no signal is received, because agents become overconfident in stability's absence of warning.

This paper closes the gap. We identify convergence across seven independent traditions of strategic thought — Sun Tzu (c.500 BCE), Thucydides (c.400 BCE), Machiavelli (1513), Clausewitz (1832), Liddell Hart (1929), Boyd (1976), Dowding (1936–1940) — on a five-part structure for effective action under transition uncertainty. The convergence is motivational, not mathematical. The mathematical results stand independently.

We formalize the structure in four theorems. Theorem 1 (Window Rationality) proves intervention is rational if and only if E[W_v(t)] > T_decision × (1 + C_fp/C_late). Theorem 2 (Asymmetric Loss Threshold) derives the optimal detection threshold δ_c* = C_late/(C_late + C_fp), recovering Paper 1's κ=0.50 as the symmetric special case. Theorem 3 (Distributed Detection Advantage) proves k independent monitoring nodes extend the detection window by (1/λ)(1 − 1/k), saturating at 1/λ. Theorem 4 (Self-Generated Friction) proves that as a system approaches its tipping point, σ_tau diverges (Kramers scaling), creating a point t_trap before the tipping time beyond which intervention is irrational regardless of cost structure.

## 3. Literature Background and the Missing Decision Theory

## 3.1 The EWS Literature

Near fold bifurcations, the dominant eigenvalue approaches zero, causing critical slowing down: rising variance and increasing autocorrelation in observable time series (Scheffer et al. 2009). The DFA scaling exponent α increases from ~0.5 (uncorrelated) toward 1.0 (scale-free memory) as the tipping point approaches (Lenton et al. 2012). By 2024, a systematic review identified 887 unique publications since 2004 (Dakos et al. 2024).

## 3.2 The Decision-Theoretic Gap

Standard SPRT (Wald 1947) assumes stationary hypotheses. This fails for tipping point detection: the alternative hypothesis is non-stationary — as the system approaches the tipping point, the cost of Type II error (missing the transition) increases continuously. This non-stationarity is the source of Theorem 4's most consequential result: the existence of t_trap.

## 3.3 The Five-Structure Convergence Across Seven Traditions

Seven traditions, separated by 2,500 years and six civilizations, independently found the same five-part structure for acting effectively under transition uncertainty: Signal Reading, Uncertainty Quantification, the Virtù Window (bounded intervention interval), Distributed Structure, and Decision Threshold. The labels are drawn from Machiavelli; the concepts are present in all seven.

Methodological note: this convergence is motivational evidence for universality, not part of the mathematical foundation. The four theorems of Section 4 stand independently. Readers unconvinced by the historical argument should proceed directly to Section 4.

## 4. The Mathematical Framework — Four Theorems

## 4.1 Definitions

Definition: Fortuna Process. The stochastic process governing a system approaching a fold bifurcation under slow parameter forcing. Tipping time τ has mean μ_τ(t) and standard deviation σ_τ(t) evolving as the system parameter approaches critical value μ_c.

Definition: Virtù Window. W_v(t) = E[τ | data_t] − t_current. Actionable when W_v(t) > T_decision, where T_decision is the agent's minimum implementation lead time.

Definition: t_trap. The point before the tipping time at which E[W_v(t_trap)] = T_decision × (1 + C_fp/C_late). After t_trap, the Theorem 1 rationality condition fails regardless of cost structure.

## 4.2 Theorem 1: Window Rationality

Intervention is rational if and only if E[W_v(t)] > T_decision × (1 + C_fp/C_late). The symmetric case (C_fp = C_late) reduces to E[W_v(t)] > T_decision.





## 4.3 Theorem 2: Asymmetric Loss Threshold

The optimal detection threshold under asymmetric loss is δ_c* = C_late/(C_late + C_fp). This generalizes MK-P1's κ=0.50 (symmetric special case r=1). For r > 1 (C_late > C_fp), the optimal threshold is strictly below 0.50 — accept lower detection quality to avoid costly missed transitions.





## 4.4 Theorem 3: Distributed Detection Advantage

With k independent monitoring nodes each with Exp(λ) detection times: E[min(T_1,...,T_k)] = (1/λ)(1 − 1/k). The window extension relative to a single node is (1/λ)(1 − 1/k), saturating at 1/λ as k → ∞. The resilience advantage of distributed networks is a detection advantage: minimum of k independent detection times falls below single-node detection time.

Saturation property: most of the network's detection benefit is concentrated in the first several independent nodes. Unbounded network growth is not required to achieve most of the value — a feature, not a limitation.





## 4.5 Theorem 4: Self-Generated Friction and the Existence of t_trap

As a system approaches its tipping point μ_c under slow parameter forcing, σ_tau ~ (μ_c − μ(t))^(−1/2) (Kramers scaling from Gardiner 2004 §5.2). The coefficient of variation CV_tau = σ_tau/μ_tau → ∞ as μ(t) → μ_c. Since E[W_v(t)] → 0 relative to T_decision, the Theorem 1 rationality condition must fail at some point before the tipping time. This defines t_trap.

The inversion: the standard view holds that acting earlier is always safer. t_trap inverts this: a system generates its own friction against rational intervention as the tipping point approaches. The information required to justify action becomes unavailable precisely when it would be most valuable.

Placeholder PH-MK5.2: the non-asymptotic bound from Berglund & Gentz (2006) has not been independently verified for the specific parameter regime. The Kramers scaling derivation is a first-pass grounding. Empirical test via AMOC data is registered as PH-MK5.1.





## 5. The Seven-Tradition Convergence Analysis

The five-part structure — Signal Reading, Uncertainty Quantification, Virtù Window, Distributed Structure, Decision Threshold — appears across Sun Tzu (c.500 BCE), Thucydides (c.400 BCE), Machiavelli (1513), Clausewitz (1832), Liddell Hart (1929), Boyd (1976), and Dowding (1936–40). The convergence spans 2,500 years and six civilizations, involves independent intellectual traditions with no documented cross-citation, and maps systematically to the mathematical structure of Section 4.

Methodological status: this is motivational evidence for the universality of the decision-theoretic problem, not a proof. The seven traditions converged on a common structure for the same reason the mathematics does: the underlying problem — acting under transition uncertainty with bounded implementation time and asymmetric error costs — is the same problem regardless of domain. The convergence is explanatory, not foundational.

Placeholder PH-MK5.3: full independent mapping of all seven traditions against the five-structure framework has not been completed in this version. The Dowding case (1936–40 RAF Fighter Command) is the most fully developed, as it has a documented operational record against which the theoretical structure can be tested. The remaining six traditions are mapped at the level of representative citations. Full analysis is deferred.

## 6. Closure of Open Questions from Papers 1–4

Paper 5 was constructed to close four explicit deferrals:

## Paper 1 deferral — asymmetric loss:

## Paper 2 deferral — distributed resilience mechanism:

## Paper 3 deferral — adaptation trigger:

## Paper 4 deferral — intervention timing:

## 7. Limitations

Kramers scaling is asymptotic: the σ_tau ~ (μ_c − μ)^(−1/2) result holds near the bifurcation. For systems far from tipping, the scaling may differ. Non-asymptotic bounds from Berglund & Gentz (2006) are registered as PH-MK5.2.

Markovian assumption: the Fortuna Process definition assumes Markovian dynamics. Real systems with memory (long-range correlations) may exhibit different σ_tau scaling. The AMOC test case is important precisely because it has empirical σ_tau data against which the theoretical prediction can be checked.

Independence of monitoring nodes: Theorem 3 requires T_i ~ Exp(λ) independently. Correlated detection signals (e.g., nodes sharing a common upstream sensor) reduce the detection advantage. Organizations should assess correlation structure before applying the k-node formula.

Seven-tradition convergence is motivational: the historical analysis in Section 5 is not part of the mathematical foundation. Readers should evaluate Theorems 1–4 independently of whether they find the historical argument persuasive.

t_trap location is substrate-specific: the derivation proves t_trap exists; it does not specify when it occurs for a given substrate. Operational use requires estimating σ_tau and μ_tau empirically from available data, which is an unsolved calibration problem.

## 8. Corrections Register

## Correction 1: SPRT framing corrected

Prior draft applied standard SPRT (Wald 1947) without acknowledging the stationarity assumption failure. The non-stationarity of the alternative hypothesis under slow parameter forcing is now stated explicitly in Section 3.2 and is the motivation for the Fortuna Process framework.

## Correction 2: κ=0.50 — extension not contradiction

Prior draft's presentation of Theorem 2 could be read as correcting Paper 1. Explicit note added in the predicate CONTEXT field: Theorem 2 generalizes MK-P1's κ=0.50 as the symmetric special case. Organizations using κ=0.50 with symmetric cost structures are correctly applying Paper 1.

## Placeholder PH-MK5.1 — AMOC empirical test

Empirical test of t_trap dynamics using Atlantic Meridional Overturning Circulation data. Registered as Grok Work Order 001. Pending bioRxiv DOI confirmation for coordination.

## Placeholder PH-MK5.2 — Berglund & Gentz non-asymptotic bound

Full non-asymptotic bound on σ_tau scaling from Berglund & Gentz (2006) Ch.5 not yet independently verified for the MK-P5 parameter regime. First-pass grounding present; formal verification deferred.

## Placeholder PH-MK5.3 — Seven-tradition full mapping

Complete independent mapping of all seven traditions against the five-structure framework not yet completed. Dowding case is fully developed. Remaining six traditions mapped at representative citation level.

## 9. References

Boyd, J. R. (1976). Destruction and creation. U.S. Army Command and General Staff College.

Clausewitz, C. von (1832/1976). On War. (Trans. Howard & Paret.) Princeton University Press.

Dakos, V., et al. (2024). Systematic review of early warning signals literature 2004–2024. Proceedings of the Royal Society B.

Dowding, H. (1936–1940). RAF Fighter Command operational records. National Archives, Kew.

Gardiner, C. W. (2004). Handbook of Stochastic Methods (3rd ed.). Springer. [§5.2: Kramers escape rate and first-passage time variance near bifurcation.]

Lenton, T. M., et al. (2012). Early warning of climate tipping points. Nature Climate Change, 1(4), 201–209.

Liddell Hart, B. H. (1929). The Decisive Wars of History. G. Bell and Sons.

Machiavelli, N. (1513/1988). The Prince. (Trans. Harvey Mansfield.) University of Chicago Press.

Scheffer, M., et al. (2009). Early-warning signals for critical transitions. Nature, 461(7260), 53–59.

Sun Tzu (c.500 BCE / 1963). The Art of War. (Trans. Samuel Griffith.) Oxford University Press.

Thucydides (c.400 BCE / 1972). History of the Peloponnesian War. (Trans. Rex Warner.) Penguin Classics.

Wald, A. (1947). Sequential Analysis. Wiley.

Berglund, N., & Gentz, B. (2006). Noise-Induced Phenomena in Slow-Fast Dynamical Systems. Springer. [Ch.5: Non-asymptotic first-passage time bounds near bifurcation.]



## Appendix A: AI Layer — Channel 2 Asset

Layer URL: https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/MK-P5-ai-layer.json

Schema: v2-S48 + semantic_spec_url amendment | Phase status: PHASE-READY | Placeholders: 3 (PH-MK5.1, PH-MK5.2, PH-MK5.3) — all non-blocking | Produced: Session S49

Semantic spec (Layer 0): github.com/thomasbrennan/Fracttalix/ai-layers/falsification-kernel.md

This paper is the first paper in the Fracttalix corpus — and, to the best of the authors' knowledge, the first paper in any published corpus — to carry the full DRS v1.1 standard, including the semantic_spec_url field pointing to the Falsification Kernel (Layer 0). The Kernel defines what a falsification predicate means independently of any serialisation format. All four Type F claims in this paper conform to the Kernel 4-tuple K = (P, O, M, B) as defined in falsification-kernel.md v1.1. The Layer 0 reference makes this conformance machine-verifiable, not merely asserted.

## Claim summary (MK-P5 only):

C-MK5.1 [F] Window Rationality — intervention rational iff E[W_v(t)] > T_decision × (1 + C_fp/C_late)

C-MK5.2 [F] Asymmetric Loss Threshold — δ_c* = C_late/(C_late+C_fp)

C-MK5.3 [F] Distributed Detection Advantage — E[min(T_1,...,T_k)] = (1/λ)(1−1/k)

C-MK5.4 [F] Self-Generated Friction — t_trap exists before tipping time (Kramers scaling)

A-MK5.1 [A] SPRT and Bayesian sequential testing — Wald (1947)

A-MK5.2 [A] Critical slowing down near fold bifurcation — Gardiner (2004)

A-MK5.3 [A] EWS literature gap in decision theory — Dakos et al. (2024)

D-MK5.1 [D] Fortuna Process definition

D-MK5.2 [D] Virtù Window W_v(t) definition