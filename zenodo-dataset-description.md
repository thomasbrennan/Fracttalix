# The Fracttalix Meta-Kaizen Series with Fracttalix Sentinel Anomaly Detection Software

**Zenodo Dataset Record — Updated Description**
*Copy-paste this into the Zenodo record description field from a desktop browser.*

---

## Fracttalix Sentinel v12.3.0

**Lightweight, regime-aware anomaly detection for time series**

Single-file Python | Zero required dependencies | `pip install fracttalix`

- Three-channel dissipative network model (structural, rhythmic, temporal)
- 38-step streaming pipeline — one observation in, rich result out (Step 0: seasonal preprocessing)
- Four signal-processing collapse indicators: maintenance burden (μ = 1−κ̄), PAC pre-cascade, diagnostic window, Kuramoto order
- Production defaults: 2.6% normal FPR (v12.3; was 35% in v12.2)
- 405 unit tests passing | Python 3.9–3.12
- GitHub: https://github.com/thomasbrennan/Fracttalix
- Software DOI: [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299)

### v12.3 Changes (FPR −93%, Mean F1 +25%)
- SeasonalPreprocessStep (Step 0): FFT deseasonalization with 10× confidence gate
- ConsensusGate: requires ≥2 soft alerts before firing (primary FPR reduction)
- VarCUSUM/CUSUM k correction: 0.5→1.0 (eliminated +0.5/step systematic bias)
- Null-distribution threshold recalibration: RFI, PE, coherence, coupling to p99 white noise
- Non-adaptive drift CUSUM: warmup-frozen baseline catches slow drift masked by EWMA

---

## Meta-Kaizen Series — 6 Papers

A measurement-theoretic governance framework derived from six axioms of conjoint measurement theory.

| Paper | Title |
|-------|-------|
| **MK-P1** | Meta-Kaizen: A General Theory and Algorithmic Framework for the Mathematical Formalization of Self-Evolving Continuous Improvement Across Arbitrary Governance Substrates |
| **MK-P2** | Meta-Kaizen in Networked Organizations: Governance Closure, Privacy Amplification, and Bayesian Calibration Under a Federated Architecture |
| **MK-P3** | The Meta-Kaizen Reasoning Network: A Formal Theory of Bisociative Question Structure, Challenge Taxonomy, and Institutional Memory Propagation |
| **MK-P4** | The Fractal Rhythm Model: Closed-Loop Governance, Regime-Aware Adaptation, and the Axiom 5 Modification for Dynamic Environments |
| **MK-P5** | On the Decision to Act: Strategic Convergence and the Mathematics of Intervention Timing at System Tipping Points *(Capstone)* |
| **MK-P6** | The Dual Reader Standard for Software: Measurement-Theoretic Falsification Applied to Executable Systems |

### Key Results

- **KVS = N × I' × C' × T** — uniquely determined scoring form from six measurement axioms (Paper 1)
- **Federated privacy architecture** — shuffle-model differential privacy with n* ≈ 100–200 minimum network size (Paper 2)
- **Bisociative question taxonomy** — exhaustive four-type challenge classification from Aristotle's predicables (Paper 3)
- **Regime-adaptive KVS** — Axiom 5' modification with FRM regime signals RDS and CSS (Paper 4)
- **Intervention timing theorems** — seven-tradition convergence on five-part decision structure; t_trap existence proof (Paper 5)

### AI Layers

Machine-readable falsification layers conforming to ai-layer-schema.json (v2-S42, Dual Reader Standard) are included for all five papers.

---

## Fractal Rhythm Model (Paper 1)

**Title:** Fracttalix Sentinel: Real-Time Streaming Anomaly Detection via a Three-Channel Dissipative Network Model

Describes association dynamics via f(t) = B + A·exp(−λt)·cos(ωt+φ) across scales from quantum decoherence to cosmological structure.

---

## Authors & License

**Authors:** Thomas Brennan · with AI collaborators Claude (Anthropic) and Grok (xAI)

**ORCID:** [0009-0002-6353-7115](https://orcid.org/0009-0002-6353-7115)

**License:** CC0 1.0 Universal (Public Domain) — no restrictions, no attribution required.
