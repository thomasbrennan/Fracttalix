# Session 59 — Independent α Extraction: Zero Free Dynamics Parameters

**Date:** 2026-03-14
**Type:** Analysis, literature compilation, parameter extraction.

---

## What Happened

### The Question (from S58)

S58 identified the remaining scholarship gap: the FRM waveform fitting showed that ω = π/(2·τ_gen) is correct (Mode B vs Mode C, Δ < 0.001), but the damping rate λ requires knowing α (bifurcation distance), which appeared to be a free parameter. Can α be extracted independently?

### The Answer: YES

For each of the 15 biological systems, the damping rate λ_obs was compiled from published time-series recordings made by domain-specific experimentalists with no knowledge of the FRM. The bifurcation distance is then computed as:

> α = −λ_obs · Γ · τ_gen

Both inputs (λ_obs from amplitude decay measurements, τ_gen from structural delay measurements) are independently published quantities. The α values range from −0.042 (KaiABC, near limit cycle) to −1.664 (glycogen supercompensation, heavily damped).

### Cross-Validation

For the 4 systems with representative time-series, independently extracted α matched Mode B fitted α within 1.6–5.9%:

| System | α_independent | α_fitted | % diff |
|--------|---------------|----------|--------|
| SCN | −0.208 | −0.202 | 2.9% |
| Xenopus | −0.390 | −0.411 | 5.3% |
| Yeast NADH | −0.260 | −0.275 | 5.9% |
| Glycogen | −1.664 | −1.692 | 1.6% |

### Regime Structure

The 15 systems cluster into three damping regimes:
- **Near-critical (|α| < 0.5):** 11 systems — circadian, metabolic, cell cycle
- **Moderate (0.5 ≤ |α| < 1.5):** 3 systems — cardiac, bone, strength
- **Heavily damped (|α| ≥ 1.5):** 1 system — glycogen supercompensation

This aligns with biological expectation: sustained oscillators (circadian, metabolic) are maintained near Hopf criticality by homeostatic mechanisms, while perturbation responses (glycogen, strength) are transient events far from bifurcation.

### Confidence Assessment

- **High confidence (5 systems):** SCN, Xenopus, cardiac APD, yeast NADH, glycogen — direct amplitude decay from clean time-series recordings
- **Medium confidence (7 systems):** KaiABC, Drosophila, Neurospora, Arabidopsis, Ca²⁺ (both), strength — measurements exist but with caveats (long recordings needed, near limit cycle, single-overshoot)
- **Low confidence (3 systems):** budding yeast, fission yeast, bone — population desynchronisation confounds single-cell damping, or limited direct data

---

## The Zero Free Parameter Claim

The FRM functional form f(t) = B + A·exp(−λt)·cos(ωt + φ) has:

**Dynamics parameters (zero free):**
- ω = π/(2·τ_gen) — from structural delay alone
- λ = λ_obs = |α|/(Γ·τ_gen) — from published damping rate

**Envelope parameters (three, fitted to initial conditions):**
- B — baseline level
- A — initial perturbation amplitude
- φ — initial phase

Compare with conventional damped oscillator models: 5+ fitted parameters (B, A, λ, ω, φ all free). The FRM reduces this to 3 fitted parameters describing initial conditions only, with all dynamics pre-specified from two independently measured quantities.

---

## Artifacts Produced

- Updated `scripts/p4_biological_validation.py`:
  - `INDEPENDENT_DAMPING_DATA` structure (15 systems with published λ_obs)
  - `compute_alpha_from_damping()` function
  - `compute_quality_factor()` function
  - `run_alpha_extraction()` analysis function with cross-check
- Updated `journal/P4-build-process.md`:
  - S59 Supplement (Sections 7.7–7.12)
  - Build Table v3.10
- Created `journal/session_59_notes.md` — this file

## Next Steps

- Prospective waveform fitting with raw published data (CircaDB for circadian, GEO for transcriptomics)
- P4 prose writing phase with all supplementary evidence integrated
- Consider adding NF-κB and p53-Mdm2 to validation set (scope decision required)
