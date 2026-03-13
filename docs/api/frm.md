# API Reference — FRMSuite

Full API reference for the FRM physics layer (`fracttalix.frm`).

---

## Import Paths

```python
from fracttalix.frm import FRMSuite, FRMSuiteResult
from fracttalix.frm import ScopeStatus, DetectorResult   # re-exported from suite.base

# Individual Layer 2 detectors (also importable directly)
from fracttalix.frm.omega import OmegaDetector
from fracttalix.frm.virtu import VirtuDetector
# Lambda is HopfDetector(method='frm'):
from fracttalix.suite import HopfDetector
```

---

## `FRMSuite`

```python
class FRMSuite:
    def __init__(
        self,
        tau_gen:          Optional[float] = None,
        hopf_ews_kwargs:  Optional[Dict]  = None,
        lambda_kwargs:    Optional[Dict]  = None,
        omega_kwargs:     Optional[Dict]  = None,
        virtu_kwargs:     Optional[Dict]  = None,
        layer1_kwargs:    Optional[Dict]  = None,
    )
```

Unified two-layer detection suite. Layer 1 is a `DetectorSuite`; Layer 2 adds FRM physics detectors that require scipy.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tau_gen` | `float or None` | `None` | FRM generation delay. Enables strong mode for Lambda and Omega. Without it, `frm_confidence` stays 0. |
| `hopf_ews_kwargs` | `dict` | `{}` | Kwargs forwarded to `HopfDetector(method='ews')` in Layer 1 |
| `lambda_kwargs` | `dict` | `{}` | Kwargs forwarded to `HopfDetector(method='frm')` — Lambda |
| `omega_kwargs` | `dict` | `{}` | Kwargs forwarded to `OmegaDetector` |
| `virtu_kwargs` | `dict` | `{}` | Kwargs forwarded to `VirtuDetector` |
| `layer1_kwargs` | `dict` | `{}` | Dict of dicts keyed by `'discord'`, `'drift'`, `'variance'`, `'coupling'`; forwarded to respective Layer 1 detectors |

**Attributes:** `tau_gen` (read-only).

### Methods

#### `update(value: float) -> FRMSuiteResult`

Feed one observation. Runs all Layer 1 and Layer 2 detectors. Returns a frozen `FRMSuiteResult`.

Layer 2 gracefully degrades: if scipy raises `ImportError` on the first call, `layer2_available` is set to `False` and all subsequent Layer 2 results return `OUT_OF_SCOPE`. Layer 1 is unaffected.

#### `reset() -> None`

Reset all detectors (Layer 1 and 2) to factory state.

#### `state_dict() -> Dict[str, Any]`

JSON-serializable snapshot. Keys: `"layer1"`, `"lambda"`, `"omega"`, `"virtu"`, `"step"`, `"layer2_available"`.

#### `load_state(sd: Dict[str, Any]) -> None`

Restore from snapshot. Use the same `tau_gen` when constructing the new instance.

---

## `FRMSuiteResult`

Frozen dataclass returned by `FRMSuite.update()`.

```python
@dataclass(frozen=True)
class FRMSuiteResult:
    layer1:              SuiteResult      # all 5 Layer 1 detector results
    lambda_:             DetectorResult   # HopfDetector(frm) — FRM damping
    omega:               DetectorResult   # OmegaDetector — FRM frequency
    virtu:               DetectorResult   # VirtuDetector — time-to-bifurcation
    frm_confidence:      int              # 0–3 (Layer 2 strong-mode alerts)
    frm_confidence_plus: int              # 0–4 (+1 if CouplingDetector alerts)
    layer2_available:    bool             # False if scipy absent
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `any_alert` | `bool` | True if any detector in Layer 1 or 2 is alerting |
| `alerts` | `list[DetectorResult]` | All currently alerting detectors (Layer 1 + 2) |

### Methods

#### `summary() -> str`

Multi-line dashboard string:

```
L1: Hopf:ok(0.02) | Disc:ok(0.05) | Drif:ok(0.00) | Vari:ok(0.01) | Coup:ok(0.03)
L2: Lam:ALERT(0.78) | Omg:ALERT(0.61) | Vrt:ok(0.42) [frm_conf=2+0] [scipy:ok]
```

#### `__iter__`

Iterates over all 8 `DetectorResult` objects in order: layer1.hopf, layer1.discord, layer1.drift, layer1.variance, layer1.coupling, lambda\_, omega, virtu.

---

## `OmegaDetector`

```python
class OmegaDetector(BaseDetector):
    def __init__(
        self,
        tau_gen:             Optional[float] = None,
        warmup:              int             = 80,
        window:              int             = 64,
        deviation_threshold: float           = 0.05,
        alert_steps:         int             = 5,
    )
```

FRM frequency integrity detector. Tests whether the observed dominant frequency matches the FRM quarter-wave theorem prediction ω = π/(2·`tau_gen`).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tau_gen` | `float or None` | `None` | Generation delay. Enables strong mode. |
| `warmup` | `int` | `80` | Observations before any verdict |
| `window` | `int` | `64` | FFT window length for frequency estimation |
| `deviation_threshold` | `float` | `0.05` | Fractional deviation \|Δω/ω_pred\| above which score increments (strong mode) |
| `alert_steps` | `int` | `5` | Consecutive above-threshold steps required before ALERT (debounce) |

**Strong mode** (`tau_gen` supplied):
- `omega_predicted = π / (2 · tau_gen)` fixed
- Uses Hann window + parabolic interpolation for sub-bin accuracy
- Smooths last 5 ω estimates before comparing to prediction
- Fires ALERT after `alert_steps` consecutive deviations
- Score = `min(1.0, consecutive_above / alert_steps)` — rises linearly

**Weak mode** (`tau_gen=None`):
- Tracks coefficient of variation (CV = std/mean) of recent ω history
- Fires ALERT when CV > 0.20 (20% frequency wander)
- Does not test FRM prediction; does not count toward `frm_confidence`

**`OUT_OF_SCOPE`** when peak-to-mean FFT magnitude ratio < 3.0 (flat spectrum / pure noise).

**`message` format (strong):** `"omega_obs=<x> omega_pred=<y> deviation=<z> consecutive=<n> mode=strong"`

**`message` format (weak):** `"omega_obs=<x> omega_mean=<y> omega_std=<z> cv=<w> mode=weak"`

### Methods

Inherits from `BaseDetector`: `update()`, `reset()`, `state_dict()`, `load_state()`.

`state_dict()` includes: `tau_gen`, `consecutive_above`, `omega_history`.

---

## `VirtuDetector`

```python
class VirtuDetector(BaseDetector):
    def __init__(
        self,
        safety_factor: float = 1.0,
        omega_trust:   bool  = True,
        warmup:        int   = 20,
    )
```

FRM time-to-bifurcation estimator. Synthesizes Lambda and Omega outputs into an estimate of the observation steps remaining before the system crosses the bifurcation.

**Formula:**

```
Δt ≈ λ / |dλ/dt|
Δt_reported = Δt / safety_factor   (Kramers correction)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `safety_factor` | `float` | `1.0` | Conservative multiplier. Values > 1 give shorter (earlier) estimates. |
| `omega_trust` | `bool` | `True` | If True, only report when OmegaDetector is in scope (FRM structure confirmed). |
| `warmup` | `int` | `20` | Minimum Lambda history steps before reporting. |

**Score:** `urgency = 1.0 − min(1.0, ttb_reported / 200.0)`. Score 0 = distant, score 1 = imminent (TTB ≤ 0).

**Confidence grades:**

| Confidence | `lam_rate` condition |
|------------|---------------------|
| `HIGH` | `lam_rate < −0.01` |
| `MEDIUM` | `lam_rate < −0.003` |
| `LOW` | otherwise |

**Status transitions:**
- `WARMUP`: step < warmup
- `NORMAL`: λ stable (lam_rate ≥ −0.001) or `time_to_bif` not available
- `OUT_OF_SCOPE`: `omega_trust=True` and omega is not in scope
- `ALERT`: urgency score ≥ `_alert_threshold` (default 0.5)

### Primary method

#### `update_frm(lambda_val, lam_rate, time_to_bif, omega_in_scope, step) -> DetectorResult`

FRMSuite-native update. Called by `FRMSuite` with pre-parsed Lambda/Omega outputs.

| Parameter | Type | Description |
|-----------|------|-------------|
| `lambda_val` | `float or None` | Current fitted λ from Lambda |
| `lam_rate` | `float` | dλ/dt (negative = declining) |
| `time_to_bif` | `float or None` | Raw TTB from Lambda: λ/\|dλ/dt\| |
| `omega_in_scope` | `bool` | Whether OmegaDetector is currently in scope |
| `step` | `int` | Current observation step |

> **Note:** `update(value)` is also available but returns score=0 with a message directing you to `update_frm()`. Use `update_frm()` when calling standalone.

**`message` format:** `"ttb=<x> confidence=<level> safety_factor=<y> omega_confirmed=<bool>"`

### Methods

Inherits from `BaseDetector`: `reset()`, `state_dict()`, `load_state()`.

`state_dict()` includes: `last_ttb`, `last_confidence`.

---

## Lambda — `HopfDetector(method='frm')`

Lambda is not a separate class; it is `HopfDetector` instantiated with `method='frm'`. See the [Suite API Reference](suite.md#hopfdetector) for full `HopfDetector` documentation.

In `FRMSuite`, Lambda is accessed as `result.lambda_` (note the trailing underscore to avoid collision with the Python keyword).

**Key FRM-mode message fields:**

```
frm λ=0.182 rate=-0.0041 ttb=44.4 omega_predicted=0.2513 mode=strong
     ^^^^^^  ^^^^^^^^^^^^  ^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^
     λ value  dλ/dt         λ/|rate|  π/(2·tau_gen)
```

---

## `frm_confidence` Computation

```python
# strong mode = tau_gen was supplied to FRMSuite
frm_confidence = 0
if lambda_.is_alert and (tau_gen is not None):
    frm_confidence += 1
if omega.is_alert and (tau_gen is not None):
    frm_confidence += 1
if virtu.is_alert:
    frm_confidence += 1

# frm_confidence_plus: independent cross-validation
frm_confidence_plus = frm_confidence + (1 if layer1.coupling.is_alert else 0)
```

`CouplingDetector` is a Layer 1 generic detector (PAC-based) that is **structurally independent** of the FRM model. Its alert is the strongest available cross-validation signal.
