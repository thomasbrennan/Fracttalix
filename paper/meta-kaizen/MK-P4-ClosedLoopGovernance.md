Meta-Kaizen Series · Paper 4 of 7

## The Fractal Rhythm Model: Closed-Loop Governance, Regime-Aware Adaptation, and the Axiom 5 Modification for Dynamic Environments

Thomas Brennan · with AI collaborators Claude (Anthropic) and Grok (xAI)

March 2026 · Submitted for peer review



## 1. Series Orientation

This is Paper 4 of five. Papers 1–3 established the axiomatic foundation, networked implementation, and reasoning architecture. This paper addresses the dynamic environment problem: what happens to Meta-Kaizen when the operating environment changes discontinuously? The FRM is an adaptive governance layer that detects regime shifts and modifies KVS computation accordingly. Paper 5 provides the decision theory for when to act on the FRM Sentinel's detections.

## 2. Abstract and AI-Reader Header

## [AI-READER HEADER] — Dual-Reader Standard, Section 2



How to verify this paper without reading it: Load the AI layer JSON at the URL above. Run the schema validator (claim_registry, inference_rule_table, phase_ready fields). Every Type F claim has a deterministic 5-part predicate — FALSIFIED IF / WHERE / EVALUATION / BOUNDARY / CONTEXT — that evaluates to FALSIFIED or NOT FALSIFIED without accessing prose. Placeholders in the register are the only unresolved dependencies.



## Abstract (Human Reader)

Papers 1–3 operate under a stationarity assumption: the substrate's operating context is stable enough that historical improvement records and calibrated thresholds remain valid. This assumption breaks under regime shifts — discontinuous changes in operating environment that render prior calibrations obsolete.

This paper introduces the Fractal Rhythm Model (FRM), an adaptive governance layer detecting two regime signals: Regime Discontinuity Score (RDS, Bayesian change-point detection) and Complexity Surge Score (CSS, volatility-normalized complexity expansion). The FRM introduces an adjusted formula KVS-hat modifying the Novelty factor during active regime signals — a deliberate, bounded departure from Paper 1's Axiom 5. Axiom 5-prime (Regime-Conditioned Essentialness) formally characterizes the departure, justifies it economically, bounds its magnitude at w_N × S_t ≤ 0.20, and specifies conditions for full restoration of original Axiom 5.

## 3. The Stationarity Problem

Three stationarity assumptions can fail discontinuously under regime shifts:

Impact calibrations from pre-shift environment do not apply to post-shift outcomes.

Novelty measurement (four-quarter rolling window) may over-penalize domains that become urgent under the new regime.

The adoption threshold κ calibrated on historical outcomes may not be appropriate for the new environment.

## 4. Regime Signal Architecture

## 4.1 Regime Discontinuity Score (RDS)

RDS_t = P(change point at t | y₁,...,y_t) computed via BOCP algorithm (Adams & MacKay 2007) with geometric run-length prior, hazard rate h = 1/L. Default L = 12 months (see calibration guidance below). RDS_t ∈ [0,1]. Threshold: RDS_t > 0.70 triggers FRM adjustment.

Calibration guidance: For quarterly-cadence implementations (standard for most Meta-Kaizen substrates), L=12 months corresponds to ~4 quarterly observations — a thin run-length prior. Recommended defaults by substrate type:

Financial governance substrates: L ≈ 16–20 quarters (4–5 years between credit cycle regime shifts)

Clinical protocol substrates with stable populations: L ≈ 12–16 quarters

Software delivery substrates subject to rapid technology change: L ≈ 8 quarters

Absent empirical history: use L=16 quarters as conservative default for quarterly-cadence implementations

## 4.2 Complexity Surge Score (CSS) — Two Formulations



## 5. Axiom 5-Prime and KVS-hat

## 5.1 Axiom 5-Prime: Regime-Conditioned Essentialness

Original Axiom 5 states that a zero-novelty candidate cannot be an improvement candidate regardless of other component values. Axiom 5-prime modifies this: when a regime signal is active (S_t = max(RDS_t, CSS_t) > 0), candidates with N_j = 0 receive an effective novelty floor w_N × S_t > 0.

Economic justification: When the environment shifts discontinuously, the fact that a solution was explored in a prior regime does not imply that its adoption decision carries forward. The regime signal is the formal mechanism for detecting exactly this condition. The prior evaluation was made under different circumstances that no longer hold.

Bound: w_N × S_t ≤ 0.20 under default weights. A standard candidate with N=0.91 always dominates a zero-novelty candidate (0.91 > 0.20).

Restoration: Setting S_t=0 or w_N=0 exactly recovers standard KVS.

## 5.2 The KVS-hat Formula

KVS-hat_j = max(w_N × S_t, N_j) × max(w_I × S_t, I'_j) × C'_j × T_j

Default weights: w_N=0.20, w_I=0.10. Behavior: when S_t=0, KVS-hat_j = KVS_j (standard). When S_t>0, zero-novelty candidates receive floor w_N × S_t > 0; candidates with N_j > w_N × S_t are unaffected.









## 6. The Extinguishing Recursion

## 6.1 Mechanism

The extinguishing recursion governs regime signal decay following successful adaptation. Let r_t denote residual regime intensity at time t, a_t ∈ {0,1} denote whether an improvement candidate was adopted in cycle t, and δ ∈ (0,1] the extinguishing rate.

r_{t+1} = r_t × (1 − δ × a_t)

Each successful improvement adoption reduces residual regime intensity by fraction δ. Passive waiting does not resolve a regime shift — only successful adaptation does.





δ_min formula: For target H_plan planning horizon and deactivation threshold ε: δ_min = 1 − (ε/r_0)^(1/H_plan). Example: H_plan=8 quarters (2 years), ε=0.10, r_0=1.0 → δ_min = 1 − 0.10^(1/8) = 0.25. Therefore δ ≥ 0.25 guarantees extinguishing within 2-year planning horizon.

## 7. Closed-Loop Architecture

The FRM operates as a closed-loop feedback system with six components: (1) substrate outcome monitoring; (2) regime signal computation (RDS and CSS); (3) KVS adjustment (S_t gates KVS vs. KVS-hat); (4) candidate evaluation; (5) principal approval; (6) outcome logging feeding back to Bayesian calibration (MK-P2 Section 6) and regime signal computation.

Domain rotation adaptation: When S_t > 0.70, the pre-committed quarterly rotation schedule shifts to on-demand scanning mode — the administrator may call an unscheduled bisociation scan within a two-week window of the regime signal.

## 8. Worked Example: IPS Under Market Regime Shift

The FRM is demonstrated using the IPS substrate under a simulated market regime shift: cross-asset correlations increasing from 0.22 to 0.61 (analogous to 2008 and 2020 correlation convergence events).



Archived candidate with N=0 (correlation-aware allocation) receives effective N = 0.20 × 0.92 = 0.18 in Q4. KVS-hat = 0.18 × 0.67 × 0.60 × 1.00 = 0.072 — below threshold but visible. After domain refreshes to N=0.91, standard KVS=0.91×0.73×0.60×1.00=0.399 applies.

## 9. Limitations

Empirical validation pending: whether RDS and CSS correctly identify real-world regime shifts has not been established. Phase 3 validation programme required.

Parameter defaults are modeling choices: w_N=0.20, w_I=0.10, δ=0.30 are operational defaults, not derived from first principles. Organizations should calibrate empirically.

BOCP hazard rate tuning: default L=12 months may cause spurious RDS triggering in substrates with high natural quarter-to-quarter variation. Substrate-specific calibration recommended.

Modular design: not every substrate experiences meaningful regime shifts. For stable-environment substrates, standard Meta-Kaizen (Papers 1–3) is the appropriate implementation.

## 10. Corrections Register

## Correction 1: CSS saturation — two formulations provided and justified

Prior draft presented only clipping formulation without acknowledging gradient loss above 1σ. Both clipping and logistic formulations now presented with trade-offs documented.

## Correction 2: Axiom 5 departure formally acknowledged and bounded

Prior draft introduced KVS-hat without acknowledging Axiom 5 violation. Axiom 5-prime now introduces the departure formally, with economic justification, magnitude bound, and restoration conditions.

## Correction 3: Asymptotic recursion fully specified

Prior draft stated "delta large enough relative to adoption rate" without quantification. Theorem 5.1 now provides δ_min = 1−(ε/r_0)^(1/H_plan) with numerical examples.

## Correction 4: BOCP hazard rate calibration guidance added

Prior draft specified L=12 months default without noting implications for quarterly-cadence implementations. Substrate-specific calibration guidance now provided.

## 11. References

Adams, R. P., & MacKay, D. J. C. (2007). Bayesian online changepoint detection. arXiv:0710.3742.

Krantz, D. H., et al. (1971). Foundations of measurement, Vol. 1. Academic Press.

Taleb, N. N. (2012). Antifragile: Things that gain from disorder. Random House.



## Appendix: AI Layer — Channel 2 Asset

Layer URL: https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/MK-P4-ai-layer.json

Schema: v2-S48 | Phase status: PHASE-READY | Placeholders: 2 (PH-MK4.1, PH-MK4.2) — both non-blocking | Produced: Session S49

Semantic spec (Layer 0): github.com/thomasbrennan/Fracttalix/ai-layers/falsification-kernel.md