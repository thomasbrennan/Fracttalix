# Fracttalix

**Real-time streaming anomaly detection for Python** — one observation at a time, no batching, no retraining.

[![PyPI](https://img.shields.io/pypi/v/fracttalix)](https://pypi.org/project/fracttalix/)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18859299.svg)](https://doi.org/10.5281/zenodo.18859299)

The library requires no external dependencies for core operation. NumPy, SciPy, Numba, Matplotlib, and tqdm are optional accelerators.

**License:** CC0-1.0 (public domain)

---

## Three APIs — Pick the Right One

| API | Best for | Install |
|-----|----------|---------|
| **`DetectorSuite`** | Domain-specific monitoring with near-zero false positives; know which anomaly type matters | `pip install fracttalix` |
| **`FRMSuite`** | Oscillatory signals with known generation delay; need time-to-bifurcation estimates | `pip install fracttalix[fast]` |
| **`SentinelDetector`** | Unknown signal types; broadest coverage; single alert/no-alert output | `pip install fracttalix` |

---

## DetectorSuite — Five Parallel Detectors

```python
from fracttalix.suite import DetectorSuite

suite = DetectorSuite()
for value in stream:
    result = suite.update(value)
    if result.any_alert:
        print(result.summary())
```

Five independent detectors (Hopf EWS, Discord, Drift, Variance, Coupling). Each reports `OUT_OF_SCOPE` when its model doesn't apply — no false consensus, honest uncertainty. Zero external dependencies.

→ [DetectorSuite How-To Guide](detector_suite.md) · [API Reference](api/suite.md)

---

## FRMSuite — FRM Physics Layer

```python
from fracttalix.frm import FRMSuite

suite = FRMSuite(tau_gen=12.5)
for value in stream:
    result = suite.update(value)
    if result.frm_confidence >= 2:
        print(f"Bifurcation signal: confidence={result.frm_confidence}")
        print(result.virtu.message)  # time-to-bifurcation estimate
```

Adds Lambda (HopfDetector frm), OmegaDetector, and VirtuDetector above the five generic detectors. `frm_confidence` (0–3) counts how many FRM physics detectors confirm the bifurcation signal. Requires numpy + scipy.

→ [FRMSuite How-To Guide](frm_suite.md) · [API Reference](api/frm.md)

---

## SentinelDetector — 37-Step Pipeline

```python
from fracttalix import SentinelDetector, SentinelConfig

det = SentinelDetector(SentinelConfig.production())
for value in my_data_stream:
    result = det.update_and_check(value)
    if result["alert"]:
        print(f"Step {result['step']}: {result['alert_reasons']}")
    dw = result.get_diagnostic_window()
    if dw["steps"] is not None:
        print(f"Time to collapse: ~{dw['steps']:.0f} steps ({dw['confidence']})")
```

37 sequential steps covering structural, rhythmic, and temporal channels. Broadest anomaly coverage with a single composite verdict. Zero external dependencies for core.

→ [Getting Started](getting_started.md) · [API Reference](api/detector.md)

---

## Installation

```bash
# Core (no external deps — all APIs available; FRMSuite Layer 2 needs scipy)
pip install fracttalix

# With numpy + scipy (unlocks FRMSuite Layer 2, accelerates FFT steps)
pip install fracttalix[fast]

# Full optional stack (adds numba JIT, matplotlib, tqdm)
pip install fracttalix[full]
```

---

## Theory

All three APIs are grounded in the **Fractal Rhythm Model** (FRM) — a Three-Channel Model of Dissipative Network Information Transmission.

**DOI:** [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299)

→ [Three-Channel Model](three_channel_model.md)
