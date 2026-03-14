# Session 60 Notes — Prospective Waveform Fitting with Real Data

**Date:** 2026-03-14
**Focus:** S60 — FRM validation against real published time-series data
**Branch:** `claude/drp8-push-p4-start-lVe5g`

---

## Objective

Close the last major gap in P4: prospective waveform fitting against real (not synthetic) experimental data. This is the strongest possible test of the FRM frequency prediction ω = π/(2·τ_gen).

## Data Sources

### 1. Neurospora circadian expression (ECHO package)
- **Origin:** Hurley et al. (2014) PNAS — clock-regulated genes in Neurospora
- **Repository:** github.com/delosh653/ECHO → DataExample.csv
- **Format:** 12 genes × 24 time points (CT2–CT48) × 3 replicates
- **Resolution:** 2-hour sampling over 48 hours
- **License:** Open source (GPL-3.0 via ECHO package)

### 2. PER2::iLuc mouse bioluminescence
- **Origin:** Per2:iLuc whole-body imaging experiment
- **Repository:** github.com/hotgly/Whole-body_Circadian
- **Format:** 1440 rows (minutes) × 25 columns (days), actogram CSV
- **Analysis window:** Days 5–18 (constant darkness), hourly binning
- **License:** Open (GitHub public repository)

## Results

### PER2::iLuc — Strongest result
- T_FRM = 24.0 hr (predicted from τ_gen = 6.0 hr)
- T_free = 23.97 hr (fitted with all 5 parameters free)
- Period error = 0.03 hr (0.1%)
- Δ(B−C) = −0.0003
- **Interpretation:** The free fit converges to the FRM-predicted frequency. The extra ω parameter adds zero predictive value.

### Neurospora — Honest mixed result
- 11/12 genes classified as circadian-range (T_free ∈ [15, 30] hr)
- Mean Δ(B−C) = −0.16 for circadian genes
- The FRM predicts T = 22.0 hr; published period is ~22.5 hr
- The 2.3% mismatch may reflect τ_gen uncertainty (condition-dependent doubling time)

## Key Methodological Point

The prospective fitting protocol ensures no circularity:
1. τ_gen is declared from published growth rates (independent of oscillation data)
2. ω is locked from τ_gen before any fitting
3. The fitting procedure sees only (t, y) pairs and the locked ω
4. Comparison with unconstrained fit tests whether the ω constraint costs fit quality

## Artifacts Produced

- Downloaded `data/raw/neurospora_echo_example.csv`
- Downloaded `data/raw/per2_iluc_actogram.csv`
- Added to `scripts/p4_biological_validation.py`:
  - `_parse_neurospora_echo()` — CSV parser
  - `_parse_per2_iluc()` — actogram parser with hourly binning
  - `fit_free_sinusoid()` — 5-parameter unconstrained fit
  - `run_prospective_fitting()` — full S60 analysis
- Updated `journal/P4-build-process.md` — Sections 7.13–7.17
- Created `journal/session_60_notes.md` — this file

## P4 Gap Closure Status

| Gap | Status |
|-----|--------|
| No prospective time-series fitting | **CLOSED (S60)** — Real data from 2 systems |
| τ_gen cherry-picking | CLOSED (S58) — 15/15 independent, 10/15 incidental |
| α as free parameter | CLOSED (S59) — extractable from published λ_obs |
| Perturbation evidence | CLOSED (S58) — 7 causal experiments |
| Independent theory | CLOSED (S58) — Novak & Tyson confirms T/τ ∈ [2, 4] |

P4 is now PHASE-COMPLETE for data analysis. Remaining: prose writing.
