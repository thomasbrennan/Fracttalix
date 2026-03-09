# Legacy Archives

Historical single-file versions of Fracttalix Sentinel, preserved for reference.
The current package is in `fracttalix/` — install with `pip install fracttalix` or `pip install -e .`.

| File | Version | Date | Lines |
|------|---------|------|-------|
| `Sentinel v 7.6` | 7.6.0 | 2026-03-04 | 53054 bytes |
| `fracttalix_sentinel_v76.py` | 7.6.0 | 2026-03-04 | 1212 |
| `fracttalix_sentinel_v77.py` | 7.7.0 | 2026-03-04 | 1441 |
| `fracttalix_sentinel_v78.py` | 7.8.0 | 2026-03-04 | 1946 |
| `fracttalix_sentinel_v79.py` | 7.9.0 | 2026-03-04 | 2876 |
| `fracttalix_sentinel_v710.py` | 7.10.0 | 2026-03-04 | 3650 |
| `fracttalix_sentinel_v711.py` | 7.11.0 | 2026-03-04 | 4139 |
| `fracttalix_sentinel_v800.py` | 8.0.0 | 2026-03-04 | 2940 |
| `fracttalix_sentinel_v900.py` | 9.0.0 | 2026-03-04 | 5954 |
| `fracttalix_sentinel_v1000.py` | 10.0.0 | 2026-03-04 | 5954 |
| `fracttalix_sentinel_v1100.py` | 11.0.0 | 2026-03-04 | 6441 |

## Version History Summary

- **v7.6 – v7.11**: Single-file detector with flat kwargs API, growing from basic
  EWMA anomaly detection to multi-channel FRM metrics.
- **v8.0**: First modular pipeline architecture; introduced `SentinelConfig` dataclass
  and 37-step pipeline.
- **v9.0**: Added frequency decomposition (Channel 2), cross-frequency coupling,
  structural-rhythmic coherence, cascade precursor, and degradation sequence logging.
- **v10.0**: Added physics-derived collapse metrics: maintenance burden μ (Tainter),
  diagnostic window Δt, PAC coefficient, Kuramoto order Φ, critical coupling κ_c.
- **v11.0**: Corrected dimensional inconsistencies in κ_c and Kuramoto Φ formulations;
  introduced phi-kappa separation, reversed-sequence (intervention) detection.
- **v12.0 (current)**: Package split into `fracttalix/` sub-modules; numpy made
  optional via `_compat` layer; `MultiStreamSentinel` added.
