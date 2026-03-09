# Fracttalix Sentinel

Fracttalix Sentinel is a pure-Python library for **real-time streaming anomaly detection**. It processes one observation at a time — no batching, no retraining — and emits a rich result on every call, including a composite anomaly score, per-channel degradation status, and an estimated time-to-collapse with confidence bounds.

The library requires no external dependencies for core operation. NumPy, SciPy, Numba, Matplotlib, and tqdm are optional accelerators.

**License:** CC0-1.0 (public domain)

## Installation

```bash
pip install fracttalix
```

## Quick Start

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

## Next Steps

- [Getting Started](getting_started.md) — installation options, factory presets, multi-stream usage, reading results
- [Three-Channel Model](three_channel_model.md) — theoretical background and physics metrics
- [API Reference](api/detector.md) — full class and method documentation
