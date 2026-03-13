# API Reference — DetectorSuite

Full API reference for the modular five-detector suite (`fracttalix.suite`).

---

## Import Paths

```python
# Full suite
from fracttalix.suite import DetectorSuite, SuiteResult

# Individual detectors
from fracttalix.suite import (
    HopfDetector,
    DiscordDetector,
    DriftDetector,
    VarianceDetector,
    CouplingDetector,
)

# Base types
from fracttalix.suite import ScopeStatus, DetectorResult, BaseDetector
```

---

## `DetectorSuite`

```python
class DetectorSuite:
    def __init__(
        self,
        hopf_kwargs:     Optional[Dict[str, Any]] = None,
        discord_kwargs:  Optional[Dict[str, Any]] = None,
        drift_kwargs:    Optional[Dict[str, Any]] = None,
        variance_kwargs: Optional[Dict[str, Any]] = None,
        coupling_kwargs: Optional[Dict[str, Any]] = None,
    )
```

Runs all five specialized detectors in parallel on a single value stream. Each detector receives the same value and produces an independent `DetectorResult`.

**Attributes:** `.hopf`, `.discord`, `.drift`, `.variance`, `.coupling` — the underlying detector instances.

### Methods

#### `update(value: float) -> SuiteResult`

Feed one observation. Returns a frozen `SuiteResult` containing all five results.

#### `reset() -> None`

Reset all detectors to factory state (clear all learned state and step counter).

#### `state_dict() -> Dict[str, Any]`

JSON-serializable snapshot of all detector states. Returns a dict with keys `"hopf"`, `"discord"`, `"drift"`, `"variance"`, `"coupling"`.

#### `load_state(sd: Dict[str, Any]) -> None`

Restore all detector states from a snapshot produced by `state_dict()`.

---

## `SuiteResult`

Frozen dataclass returned by `DetectorSuite.update()`.

```python
@dataclass(frozen=True)
class SuiteResult:
    hopf:     DetectorResult
    discord:  DetectorResult
    drift:    DetectorResult
    variance: DetectorResult
    coupling: DetectorResult
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `any_alert` | `bool` | True if at least one detector is in ALERT status |
| `alerts` | `list[DetectorResult]` | Detectors currently in ALERT |
| `in_scope` | `list[DetectorResult]` | Detectors in NORMAL or ALERT (not WARMUP/OOS) |
| `out_of_scope` | `list[DetectorResult]` | Detectors that declared OUT_OF_SCOPE |

### Methods

#### `summary() -> str`

One-line dashboard string, e.g.:

```
Hopf:ok(0.02) | Disc:ok(0.08) | Drif:ALERT(0.73) | Vari:ok(0.01) | Coup:OOS
```

#### `__iter__`

Iterates over all five `DetectorResult` objects in order: hopf, discord, drift, variance, coupling.

---

## `DetectorResult`

Frozen dataclass returned by each individual detector's `update()` call.

```python
@dataclass(frozen=True)
class DetectorResult:
    detector: str         # detector class name
    status:   ScopeStatus # WARMUP | NORMAL | ALERT | OUT_OF_SCOPE
    score:    float       # anomaly score 0.0–1.0
    message:  str         # diagnostic detail string
    step:     int         # observation step number
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `is_alert` | `bool` | True iff `status == ScopeStatus.ALERT` |
| `in_scope` | `bool` | True iff `status in (NORMAL, ALERT)` |

---

## `ScopeStatus`

```python
class ScopeStatus(Enum):
    WARMUP       = "warmup"        # Not enough data yet for a verdict
    NORMAL       = "normal"        # In scope; no anomaly
    ALERT        = "alert"         # Anomaly detected
    OUT_OF_SCOPE = "out_of_scope"  # Detector model doesn't apply
```

---

## `HopfDetector`

```python
class HopfDetector(BaseDetector):
    def __init__(
        self,
        method:         str   = 'ews',  # 'ews' or 'frm'
        warmup:         int   = 50,
        window:         int   = 40,
        var_threshold:  float = 0.6,
        ac_threshold:   float = 0.6,
        # frm-mode only:
        tau_gen:        Optional[float] = None,
        fit_interval:   int   = 5,
    )
```

**EWS mode** (`method='ews'`): Fires when rolling variance or lag-1 autocorrelation exceeds `var_threshold` or `ac_threshold` respectively, relative to warmup baseline. Null FPR target: 0% on N(0,1).

**FRM mode** (`method='frm'`): Curve-fits amplitude envelope to FRM exponential model, tracks damping λ. Requires scipy + numpy. Used as `Lambda` in `FRMSuite`. See [FRMSuite API reference](frm.md).

**`message` format (EWS):** `"ews var_ratio=<x> ac_ratio=<y> var_alert=<bool> ac_alert=<bool>"`

**`message` format (FRM):** `"frm λ=<x> rate=<y> ttb=<z> omega_predicted=<w> mode=strong|weak"`

---

## `DiscordDetector`

```python
class DiscordDetector(BaseDetector):
    def __init__(
        self,
        window:     int   = 10,
        warmup:     int   = 50,
        threshold:  float = 3.0,
    )
```

Detects point and contextual anomalies using z-normalized Euclidean distance to nearest non-self-match neighbor in the recent window. Fires ALERT when the discord distance exceeds `threshold`. Reports `OUT_OF_SCOPE` until there is sufficient history to build a comparison pool (requires at least `2 × window + warmup` observations).

**`message` format:** `"discord distance=<x> threshold=<y>"`

---

## `DriftDetector`

```python
class DriftDetector(BaseDetector):
    def __init__(
        self,
        warmup:    int   = 50,
        cusum_k:   float = 0.5,
        cusum_h:   float = 8.0,
        ph_delta:  float = 0.01,
        ph_lambda: float = 50.0,
    )
```

Detects slow mean drift using non-adaptive CUSUM (reference mean frozen after warmup) and the Page-Hinkley test. The frozen reference prevents the detector from adapting away from an ongoing drift.

- `cusum_k`: allowance in σ units (typically 0.5σ)
- `cusum_h`: CUSUM decision threshold
- `ph_delta`: Page-Hinkley minimum magnitude
- `ph_lambda`: Page-Hinkley threshold

Fires when either CUSUM or Page-Hinkley crosses its threshold. **`message` format:** `"drift cusum_hi=<x> cusum_lo=<y> ph=<z>"`

---

## `VarianceDetector`

```python
class VarianceDetector(BaseDetector):
    def __init__(
        self,
        warmup:   int   = 50,
        cusum_k:  float = 1.0,
        cusum_h:  float = 10.0,
    )
```

Detects sudden volatility changes using CUSUM on z² (squared z-score). Under the null N(0,1), E[z²] = 1.0; `cusum_k = 1.0` is calibrated to this baseline.

- A sustained increase in z² → `cusum_hi` crosses `cusum_h` → ALERT
- A sustained decrease in z² → `cusum_lo` crosses `cusum_h` → ALERT (variance collapse)

**`message` format:** `"variance cusum_hi=<x> cusum_lo=<y> z_sq=<z>"`

---

## `CouplingDetector`

```python
class CouplingDetector(BaseDetector):
    def __init__(
        self,
        warmup:     int   = 80,
        window:     int   = 64,
        threshold:  float = 0.24,
    )
```

Tracks cross-frequency Phase-Amplitude Coupling (PAC) via FFT decomposition. Fires when the composite coupling score drops below `threshold` for a sustained period. Reports `OUT_OF_SCOPE` when the spectrum is too flat (no dominant frequency) or the window is too small.

**`message` format:** `"coupling score=<x> threshold=<y>"`

---

## `BaseDetector`

Abstract base class for all detectors.

```python
class BaseDetector(ABC):
    def __init__(self, name: str, warmup: int, window_size: int)

    def update(self, value: float) -> DetectorResult  # concrete implementation via template
    def reset(self) -> None                            # abstract
    def state_dict(self) -> Dict[str, Any]
    def load_state(self, sd: Dict[str, Any]) -> None
```

Subclasses implement `_check_scope(window)` and `_compute(window)`. The base class handles warmup, window management, and `DetectorResult` construction.

**Alert threshold:** `self._alert_threshold` (default 0.5). `_compute()` returns a score in [0, 1]; ALERT fires when score ≥ threshold.
