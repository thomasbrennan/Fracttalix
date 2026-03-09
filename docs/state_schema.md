# State Schema

Fracttalix Sentinel supports full state serialization and deserialization via `save_state()` / `load_state()`. This enables crash recovery, deployment handoff, and reproducible replay.

## Saving and Loading State

```python
# Save
json_str = det.save_state()

# Restore on a fresh detector with the same config
det2 = SentinelDetector(cfg)
det2.load_state(json_str)
```

For `MultiStreamSentinel`:

```python
state_json = mss.save_all()
mss.load_all(state_json)
```

`save_state()` returns a UTF-8 JSON string. The string is self-contained: it includes the schema version, the config snapshot, and all per-step state dictionaries.

## JSON Structure

```json
{
  "schema_version": "12",
  "step_count": 1500,
  "config": {
    "alpha": 0.1,
    "dev_alpha": 0.1,
    "multiplier": 3.0,
    "warmup_periods": 30,
    "regime_threshold": 3.5,
    "regime_alpha_boost": 2.0,
    "regime_boost_decay": 0.9,
    "multivariate": false,
    "n_channels": 1,
    "cov_alpha": 0.05,
    "rpi_window": 64,
    "rfi_window": 64,
    "rpi_threshold": 0.6,
    "rfi_threshold": 0.4,
    "pe_order": 3,
    "pe_window": 50,
    "pe_threshold": 0.05,
    "ews_window": 40,
    "ews_threshold": 0.6,
    "sti_window": 20,
    "tps_window": 30,
    "osc_damp_window": 20,
    "osc_threshold": 1.5,
    "cpd_window": 30,
    "cpd_threshold": 2.0,
    "ph_delta": 0.01,
    "ph_lambda": 50.0,
    "cusum_k": 0.5,
    "cusum_h": 5.0,
    "var_cusum_k": 0.5,
    "var_cusum_h": 5.0,
    "alert_cooldown_steps": 0,
    "seasonal_period": 0,
    "quantile_threshold_mode": false,
    "aqb_window": 200,
    "aqb_q_low": 0.01,
    "aqb_q_high": 0.99,
    "history_maxlen": 5000,
    "csv_path": "",
    "log_level": "WARNING",
    "enable_frequency_decomposition": true,
    "min_window_for_fft": 32,
    "enable_coupling_detection": true,
    "coupling_degradation_threshold": 0.3,
    "coupling_trend_window": 10,
    "enable_channel_coherence": true,
    "coherence_threshold": 0.4,
    "coherence_window": 20,
    "enable_cascade_detection": true,
    "cascade_ews_threshold": 2,
    "enable_sequence_logging": true,
    "sequence_retention": 1000,
    "warn_on_numpy_fallback": true
  },
  "steps": {
    "CoreEWMAStep": {
      "ewma": 42.17,
      "dev_ewma": 1.83,
      "initialized": true
    },
    "StructuralSnapshotStep": {
      "window": [41.2, 42.0, 43.1, 42.8, 41.9]
    },
    "FrequencyDecompositionStep": {
      "window": [41.2, 42.0, 43.1, 42.8, 41.9],
      "band_history": []
    },
    "CUSUMStep": {
      "cusum_pos": 0.0,
      "cusum_neg": 0.0
    },
    "RegimeStep": {
      "boost_remaining": 0.0
    },
    "VarCUSUMStep": {
      "cusum_var_pos": 0.0,
      "cusum_var_neg": 0.0
    },
    "PageHinkleyStep": {
      "ph_sum": 12.4,
      "ph_min": 0.0,
      "ph_n": 1500
    },
    "ThroughputEstimationStep": {
      "coupling_history": [0.71, 0.69, 0.72, 0.70, 0.68]
    },
    "PACCoefficientStep": {
      "pac_history": [0.38, 0.36, 0.35, 0.34, 0.33]
    },
    "KuramotoOrderStep": {
      "phi_history": [0.81, 0.79, 0.80, 0.77, 0.75]
    },
    "SequenceOrderingStep": {
      "sequence_log": ["COUPLING_FIRST", "STABLE", "STABLE"]
    }
  }
}
```

The `steps` object contains one key per step class name. Steps that maintain no persistent state (e.g., purely stateless transformers) may omit their entry or include an empty object.

## Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Major version of the schema; `"12"` for V12 |
| `step_count` | int | Total observations processed so far |
| `config` | object | Full snapshot of the `SentinelConfig` used at save time |
| `steps` | object | Per-step state dictionaries, keyed by class name |

## Version Compatibility

Breaking changes between major versions are noted in `CHANGELOG.md`. The `schema_version` field allows the loader to detect mismatches and apply migrations.

| Schema Version | Library Version | Notes |
|----------------|-----------------|-------|
| `"12"` | V12.x | Current format; adds V12 physics step state |
| `"11"` | V11.x (legacy) | Missing `ThroughputEstimationStep`, `PACCoefficientStep`, `KuramotoOrderStep`, `SequenceOrderingStep` step keys |
| `"10"` | V10.x (legacy) | Alias keys (`rsi`) present in step state |

## Migrating State from V11 to V12

V12 added four new step-state keys in the `steps` object:

- `ThroughputEstimationStep` — `coupling_history`
- `PACCoefficientStep` — `pac_history`
- `KuramotoOrderStep` — `phi_history`
- `SequenceOrderingStep` — `sequence_log`

V11 state files do not contain these keys. When `load_state()` is called with a V11 state blob, the missing step states are silently initialized to their defaults (empty histories). The detector will behave as if those steps have just been initialized, requiring a short re-warmup period (typically `coupling_trend_window` = 10 observations) before the physics-group metrics stabilize.

To migrate explicitly:

```python
import json

with open("sentinel_v11_state.json") as f:
    state = json.load(f)

# Inject default V12 step states if missing
state.setdefault("schema_version", "12")
state["steps"].setdefault("ThroughputEstimationStep", {"coupling_history": []})
state["steps"].setdefault("PACCoefficientStep", {"pac_history": []})
state["steps"].setdefault("KuramotoOrderStep", {"phi_history": []})
state["steps"].setdefault("SequenceOrderingStep", {"sequence_log": []})

det = SentinelDetector(cfg)
det.load_state(json.dumps(state))
```

## V10 Alias Key Removal

V11 removed the alias key `rsi` from `SSIStep` state (the key was a V10 compatibility shim mapping the old `rsi` scratch name to the canonical `ssi` name). V12 does not carry this alias. If you are migrating from V10 state that contains `"rsi"` in `SSIStep`, remove the key before loading:

```python
state["steps"].get("SSIStep", {}).pop("rsi", None)
```

## MultiStreamSentinel State Format

`mss.save_all()` returns a JSON object where each key is a stream ID and the value is the single-stream state blob described above:

```json
{
  "sensor_42": { "schema_version": "12", "step_count": 1500, ... },
  "sensor_99": { "schema_version": "12", "step_count": 320, ... }
}
```

`mss.load_all(json_str)` restores all streams. Streams present in the JSON but not currently active in the `MultiStreamSentinel` are created automatically.
