# fracttalix/suite/hopf.py
# HopfDetector — Pre-transition early warning via critical slowing down.
#
# method='ews' (default, no scipy needed):
#   Theorem basis (P1 / EWS):
#     Near a Hopf bifurcation (or any fold/transcritical transition), the system
#     loses its ability to recover from perturbations.  This manifests as:
#       1. Rising variance (the noise is amplified more).
#       2. Rising lag-1 autocorrelation (recovery takes longer → memory builds).
#     Both signals rise before the transition, not at it — that's the early warning.
#
# method='frm' (requires scipy + numpy):
#   Theorem basis (FRM / Lambda):
#     The Fractal Rhythm Model derives a damped oscillation form:
#       f(t) = B + A·exp(-λt)·cos(ωt + φ),  ω = π/(2·τ_gen)
#     When λ → 0 the system approaches its Hopf bifurcation.  Fits the model
#     to streaming data, tracks λ over time, and estimates time-to-bifurcation
#     as Δt = λ / |dλ/dt|.  More precise than EWS for FRM-shaped signals;
#     also provides tau_gen_implied and time_to_transition estimates.
#     Validated: Melbourne temperature FPR 8.0% (vs 66.6% pre-fix).
#
# OUT_OF_SCOPE conditions (EWS):
#   • White noise: AC(1) is persistently near zero → slowing-down signal
#     is indistinguishable from baseline; report nothing.
#   • Sustained mean shift: the signal has already jumped → that's a regime
#     change, not an approach to one.  DiscordDetector / DriftDetector see it.
#   • Variance already high at warmup end: we have no pre-transition baseline
#     to compare against → insufficient scope.
#
# OUT_OF_SCOPE conditions (FRM):
#   • Fit fails to converge or R² < threshold → data is not FRM-shaped.
#   • LIMIT_CYCLE: λ ≈ 0 with no declining trend → sustained oscillation,
#     not a damped system approaching bifurcation.
#
# Best at: oscillatory or autocorrelated signals approaching a qualitative
#          state change (oscillation onset, equilibrium loss, phase transition).
# Mediocre at: step functions (reports OUT_OF_SCOPE after the jump).
# Useless at: pure white noise (always OUT_OF_SCOPE by design).

import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from fracttalix.suite.base import (
    BaseDetector, DetectorResult, ScopeStatus,
    _mean, _variance, _std, _ac1,
)

# FRM universal constant: Γ = 1 + π²/4 ≈ 3.467 (loop impedance)
_GAMMA = 1.0 + math.pi ** 2 / 4.0


def _frm_estimate_omega(data):
    """Estimate dominant angular frequency from FFT peak."""
    import numpy as np
    n = len(data)
    centered = data - np.mean(data)
    fft_vals = np.abs(np.fft.rfft(centered))
    if len(fft_vals) <= 1:
        return 0.1
    peak_idx = int(np.argmax(fft_vals[1:])) + 1
    omega = 2.0 * math.pi * peak_idx / n
    return max(omega, 0.01)


def _frm_fit_lambda(data, omega_fixed):
    """Fit FRM model with fixed ω; return (lam, r2, converged, prev_params)."""
    import numpy as np
    from scipy.optimize import curve_fit

    n = len(data)
    t = np.arange(n, dtype=float)

    B_init = float(np.mean(data[-max(1, n // 10):]))
    A_init = float(np.max(np.abs(data - B_init))) or 1.0

    # Rough lambda from envelope decay
    abs_vals = np.abs(data - np.mean(data))
    peaks = [(i, abs_vals[i]) for i in range(1, n - 1)
             if abs_vals[i] > abs_vals[i - 1] and abs_vals[i] > abs_vals[i + 1]]
    lam_init = 0.1
    if len(peaks) >= 2:
        pt = np.array([p[0] for p in peaks], dtype=float)
        pv = np.array([p[1] for p in peaks], dtype=float)
        mask = pv > 0
        if mask.sum() >= 2:
            lv = np.log(pv[mask])
            tm = pt[mask]
            denom = np.sum((tm - np.mean(tm)) ** 2)
            if denom > 1e-12:
                slope = np.sum((tm - np.mean(tm)) * (lv - np.mean(lv))) / denom
                lam_init = max(-slope, 0.0)

    def model(t_arr, B, A, lam, phi):
        return B + A * np.exp(-lam * t_arr) * np.cos(omega_fixed * t_arr + phi)

    try:
        popt, _ = curve_fit(
            model, t, data,
            p0=[B_init, A_init, lam_init, 0.0],
            bounds=([-np.inf, -np.inf, 0.0, -2 * math.pi],
                    [np.inf, np.inf, 50.0, 2 * math.pi]),
            maxfev=2000,
        )
        B_fit, A_fit, lam_fit, phi_fit = popt
        y_pred = model(t, *popt)
        ss_res = np.sum((data - y_pred) ** 2)
        ss_tot = np.sum((data - np.mean(data)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return lam_fit, float(r2), True
    except (RuntimeError, ValueError):
        return lam_init, 0.0, False


class HopfDetector(BaseDetector):
    """Detect pre-transition early warning via critical slowing down.

    Parameters
    ----------
    method : str
        'ews' (default) — generic EWS: rising variance + AC(1), no scipy.
        'frm' — FRM model fitting: track λ → 0, requires scipy + numpy.
        Both detect Hopf bifurcations; FRM additionally gives time-to-transition
        estimates and is validated for FRM-shaped oscillatory signals.
    warmup : int
        Observations before any verdict (default 60).  Must be long enough
        to establish a stable variance+AC(1) baseline.
    window : int
        Rolling window for current variance/AC(1) computation (default 40).
    ews_threshold : float
        Score threshold above which status = ALERT (default 0.55).
    ac1_min : float
        Minimum baseline AC(1) for the signal to be in scope.  Signals with
        AC(1) < ac1_min are treated as white noise → OUT_OF_SCOPE (default 0.1).
    mean_shift_z : float
        If |current_mean − warmup_mean| > mean_shift_z * warmup_std the signal
        has already jumped; report OUT_OF_SCOPE (default 3.5).
    tau_gen : float or None
        FRM method only. If provided, fixes ω = π/(2·tau_gen) for fitting.
        If None (default), ω is estimated from FFT peak each window.
    frm_r2_min : float
        FRM method only. Minimum R² for fit to be considered in scope (default 0.30).
    frm_fit_interval : int
        FRM method only. Run scipy fit every N observations (default 5).
    frm_lambda_window : int
        FRM method only. History length for dλ/dt estimation (default 20).
    """

    def __init__(
        self,
        method: str = 'ews',
        warmup: int = 60,
        window: int = 40,
        ews_threshold: float = 0.55,
        ac1_min: float = 0.10,
        mean_shift_z: float = 3.5,
        tau_gen: Optional[float] = None,
        frm_r2_min: float = 0.30,
        frm_fit_interval: int = 5,
        frm_lambda_window: int = 20,
    ):
        if method not in ('ews', 'frm'):
            raise ValueError(f"method must be 'ews' or 'frm', got {method!r}")
        self._method = method
        super().__init__("HopfDetector", warmup=warmup, window_size=max(window, warmup))
        self._ews_window = window
        self._ews_threshold = ews_threshold
        self._ac1_min = ac1_min
        self._mean_shift_z = mean_shift_z
        self._alert_threshold = ews_threshold

        # Warmup baseline (frozen after warmup)
        self._warmup_var: float = 0.0
        self._warmup_ac1: float = 0.0
        self._warmup_mean: float = 0.0
        self._warmup_std: float = 1.0
        self._baseline_set: bool = False

        # EWMA trackers for trend detection
        self._var_ewma: float = 0.0
        self._ac1_ewma: float = 0.0

        # FRM method state
        self._tau_gen = tau_gen
        self._frm_r2_min = frm_r2_min
        self._frm_fit_interval = frm_fit_interval
        self._frm_lambda_window = frm_lambda_window
        self._frm_lambda_history: deque = deque(maxlen=frm_lambda_window)
        self._frm_fit_counter: int = 0
        self._frm_last_lam: Optional[float] = None
        self._frm_last_scope: str = "INSUFFICIENT_DATA"

    # ------------------------------------------------------------------
    # FRM method: override update() entirely
    # ------------------------------------------------------------------

    def update(self, value: float) -> DetectorResult:
        if self._method == 'frm':
            return self._frm_update(value)
        return super().update(value)

    def _frm_update(self, value: float) -> DetectorResult:
        """FRM-specific update: fit damped oscillation, track λ → 0."""
        self._window.append(float(value))
        step = self._step
        self._step += 1

        if step < self._warmup:
            return DetectorResult(
                detector=self._name,
                status=ScopeStatus.WARMUP,
                score=0.0,
                message=f"warmup ({step}/{self._warmup})",
                step=step,
            )

        self._frm_fit_counter += 1
        data_list = list(self._window)

        # Only run scipy fit every N steps
        if self._frm_fit_counter % self._frm_fit_interval == 0 or self._frm_last_lam is None:
            import numpy as np
            data = np.array(data_list, dtype=float)

            if self._tau_gen is not None and self._tau_gen > 0:
                omega = math.pi / (2.0 * self._tau_gen)
            else:
                omega = _frm_estimate_omega(data)

            lam, r2, converged = _frm_fit_lambda(data, omega)

            if not converged or r2 < self._frm_r2_min:
                self._frm_last_scope = "OUT_OF_SCOPE"
            elif lam * len(data_list) < 0.5:
                # Less than 39% amplitude decay → sustained oscillation (limit cycle)
                self._frm_last_scope = "LIMIT_CYCLE"
            else:
                self._frm_last_scope = "IN_SCOPE"
                self._frm_last_lam = lam
                self._frm_lambda_history.append(lam)

            tau_implied = math.pi / (2.0 * omega) if omega > 1e-10 else None
        else:
            r2 = 0.0
            tau_implied = None

        scope = self._frm_last_scope
        if scope in ("OUT_OF_SCOPE", "LIMIT_CYCLE"):
            return DetectorResult(
                detector=self._name,
                status=ScopeStatus.OUT_OF_SCOPE,
                score=0.0,
                message=f"frm scope={scope}",
                step=step,
            )

        lam, lam_rate, time_to_bif = self._frm_lambda_trend()

        # Require actively declining λ (lam_rate < -0.001) for any alert
        if lam is None or lam_rate >= -1e-3:
            return DetectorResult(
                detector=self._name,
                status=ScopeStatus.NORMAL,
                score=0.0,
                message=f"frm λ={lam:.4f} rate={lam_rate:.5f} stable" if lam is not None else "frm insufficient_history",
                step=step,
            )

        # Score: 0→1 based on how close λ is to zero, weighted by trend strength
        lam_score = min(1.0, max(0.0, 1.0 - lam / 0.5))
        rate_score = min(1.0, max(0.0, abs(lam_rate) / 0.05))
        score = 0.6 * lam_score + 0.4 * rate_score

        ttb_str = f" ttb={time_to_bif:.1f}" if time_to_bif is not None else ""
        msg = f"frm λ={lam:.4f} rate={lam_rate:.5f}{ttb_str}"
        status = ScopeStatus.ALERT if score >= self._alert_threshold else ScopeStatus.NORMAL
        return DetectorResult(
            detector=self._name,
            status=status,
            score=score,
            message=msg,
            step=step,
        )

    def _frm_lambda_trend(self) -> Tuple[Optional[float], float, Optional[float]]:
        """Return (current_lam, lam_rate, time_to_bifurcation) from history."""
        history = list(self._frm_lambda_history)
        if len(history) < 3:
            return None, 0.0, None

        import numpy as np
        lams = np.array(history, dtype=float)
        t = np.arange(len(lams), dtype=float)
        t_mean = np.mean(t)
        lam_mean = np.mean(lams)
        numer = np.sum((t - t_mean) * (lams - lam_mean))
        denom = np.sum((t - t_mean) ** 2)
        lam_rate = float(numer / denom) if abs(denom) > 1e-12 else 0.0

        current_lam = float(lams[-1])
        time_to_bif = None
        if lam_rate < -1e-8 and current_lam > 0:
            time_to_bif = current_lam / abs(lam_rate) * self._frm_fit_interval

        return current_lam, lam_rate, time_to_bif

    # ------------------------------------------------------------------
    # EWS method (default)
    # ------------------------------------------------------------------

    def _set_baseline(self, window: List[float]) -> None:
        self._warmup_mean = _mean(window)
        self._warmup_std = max(_std(window), 1e-10)
        self._warmup_var = _variance(window)
        self._warmup_ac1 = _ac1(window)
        self._var_ewma = self._warmup_var
        self._ac1_ewma = self._warmup_ac1
        self._baseline_set = True

    def _check_scope(self, window: List[float]) -> bool:
        if not self._baseline_set:
            self._set_baseline(window[-self._ews_window:])

        recent = window[-self._ews_window:]

        # Scope gate 1: baseline AC(1) too low → white noise
        if self._warmup_ac1 < self._ac1_min:
            return False

        # Scope gate 2: current mean has shifted beyond mean_shift_z
        cur_mean = _mean(recent)
        z_shift = abs(cur_mean - self._warmup_mean) / self._warmup_std
        if z_shift > self._mean_shift_z:
            return False

        return True

    def _compute(self, window: List[float]) -> Tuple[float, str]:
        recent = window[-self._ews_window:]
        alpha = 0.15

        cur_var = _variance(recent)
        cur_ac1 = _ac1(recent)

        self._var_ewma = alpha * cur_var + (1 - alpha) * self._var_ewma
        self._ac1_ewma = alpha * cur_ac1 + (1 - alpha) * self._ac1_ewma

        # Rising variance: how much has variance grown relative to warmup?
        var_ratio = cur_var / (self._warmup_var + 1e-10)
        # Map: 1.0 → 0.0 (no change), 4.0 → 1.0 (4× the baseline variance)
        var_score = min(1.0, max(0.0, (var_ratio - 1.0) / 3.0))

        # Rising AC(1): how much has it grown relative to warmup?
        ac1_delta = cur_ac1 - self._warmup_ac1
        # +0.3 delta → score = 1.0; negative → score = 0.0
        ac1_score = min(1.0, max(0.0, ac1_delta / 0.30))

        # EWS score: both must be rising for a true Hopf precursor
        ews_score = 0.5 * var_score + 0.5 * ac1_score

        if ews_score >= self._ews_threshold * 1.4:
            regime = "critical"
        elif ews_score >= self._ews_threshold:
            regime = "approaching"
        else:
            regime = "stable"

        msg = (
            f"ews={ews_score:.3f} var_ratio={var_ratio:.2f} "
            f"ac1={cur_ac1:.3f}(base={self._warmup_ac1:.3f}) regime={regime}"
        )
        return ews_score, msg

    def reset(self) -> None:
        super().reset()
        self._baseline_set = False
        self._warmup_var = 0.0
        self._warmup_ac1 = 0.0
        self._warmup_mean = 0.0
        self._warmup_std = 1.0
        self._var_ewma = 0.0
        self._ac1_ewma = 0.0
        self._frm_lambda_history.clear()
        self._frm_fit_counter = 0
        self._frm_last_lam = None
        self._frm_last_scope = "INSUFFICIENT_DATA"

    def state_dict(self) -> Dict[str, Any]:
        sd = super().state_dict()
        sd.update({
            "method": self._method,
            "baseline_set": self._baseline_set,
            "warmup_var": self._warmup_var,
            "warmup_ac1": self._warmup_ac1,
            "warmup_mean": self._warmup_mean,
            "warmup_std": self._warmup_std,
            "var_ewma": self._var_ewma,
            "ac1_ewma": self._ac1_ewma,
            "frm_lambda_history": list(self._frm_lambda_history),
            "frm_fit_counter": self._frm_fit_counter,
            "frm_last_lam": self._frm_last_lam,
            "frm_last_scope": self._frm_last_scope,
        })
        return sd

    def load_state(self, sd: Dict[str, Any]) -> None:
        super().load_state(sd)
        self._baseline_set = sd.get("baseline_set", False)
        self._warmup_var = sd.get("warmup_var", 0.0)
        self._warmup_ac1 = sd.get("warmup_ac1", 0.0)
        self._warmup_mean = sd.get("warmup_mean", 0.0)
        self._warmup_std = sd.get("warmup_std", 1.0)
        self._var_ewma = sd.get("var_ewma", 0.0)
        self._ac1_ewma = sd.get("ac1_ewma", 0.0)
        self._frm_lambda_history = deque(
            sd.get("frm_lambda_history", []), maxlen=self._frm_lambda_window
        )
        self._frm_fit_counter = sd.get("frm_fit_counter", 0)
        self._frm_last_lam = sd.get("frm_last_lam", None)
        self._frm_last_scope = sd.get("frm_last_scope", "INSUFFICIENT_DATA")
