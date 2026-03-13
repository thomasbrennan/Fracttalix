# fracttalix/steps/hopf.py
# HopfDetectorStep — fit FRM damped oscillation form, track λ → 0
#
# The Fractal Rhythm Model derives:
#   f(t) = B + A·exp(-λt)·cos(ωt + φ)
#   ω = π/(2·τ_gen)       (Hopf quarter-wave theorem)
#   λ = |α|/(Γ·τ_gen)     (perturbation expansion, Γ = 1 + π²/4)
#
# When λ → 0 the system approaches its Hopf bifurcation.
# This step fits the model to streaming data and watches for that.

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional

from fracttalix.config import SentinelConfig
from fracttalix.steps.base import DetectorStep
from fracttalix.window import StepContext

# FRM universal constants
GAMMA = 1.0 + math.pi**2 / 4.0  # ≈ 3.467 — loop impedance constant


def _frm_model(t, B, A, lam, phi, omega):
    """FRM functional form: f(t) = B + A·exp(-λt)·cos(ωt + φ)."""
    import numpy as np

    return B + A * np.exp(-lam * t) * np.cos(omega * t + phi)


def _estimate_omega_from_fft(data):
    """Estimate dominant frequency from FFT peak."""
    import numpy as np

    n = len(data)
    centered = data - np.mean(data)
    fft_vals = np.abs(np.fft.rfft(centered))
    # Skip DC component (index 0)
    if len(fft_vals) <= 1:
        return 0.1  # fallback
    peak_idx = np.argmax(fft_vals[1:]) + 1
    omega = 2.0 * math.pi * peak_idx / n
    return max(omega, 0.01)  # avoid zero


def _estimate_lambda_from_envelope(data):
    """Estimate decay rate from log of peak envelope."""
    import numpy as np

    n = len(data)
    centered = data - np.mean(data)
    abs_vals = np.abs(centered)

    # Find local maxima (peaks)
    peaks = []
    for i in range(1, n - 1):
        if abs_vals[i] > abs_vals[i - 1] and abs_vals[i] > abs_vals[i + 1]:
            peaks.append((i, abs_vals[i]))

    if len(peaks) < 2:
        return 0.1  # fallback

    peak_times = np.array([p[0] for p in peaks], dtype=float)
    peak_vals = np.array([p[1] for p in peaks], dtype=float)

    # Log-linear fit: log(envelope) = -λt + const
    mask = peak_vals > 0
    if mask.sum() < 2:
        return 0.1
    log_vals = np.log(peak_vals[mask])
    t_vals = peak_times[mask]

    # Simple OLS
    t_mean = np.mean(t_vals)
    log_mean = np.mean(log_vals)
    numer = np.sum((t_vals - t_mean) * (log_vals - log_mean))
    denom = np.sum((t_vals - t_mean) ** 2)
    if abs(denom) < 1e-12:
        return 0.1
    slope = numer / denom
    return max(-slope, 0.0)  # λ ≥ 0


class HopfDetectorStep(DetectorStep):
    """Fit FRM damped oscillation and track decay rate λ toward zero.

    When λ → 0, the system approaches its Hopf bifurcation (critical
    transition). This step fits f(t) = B + A·exp(-λt)·cos(ωt+φ) to a
    sliding window using nonlinear least squares, tracks λ over time,
    and estimates time-to-bifurcation as Δt = λ / |dλ/dt|.

    Requires scipy. Raises ImportError on first update() if unavailable.
    """

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._window: deque = deque(maxlen=self.cfg.hopf_fit_window)
        self._lambda_history: deque = deque(
            maxlen=self.cfg.hopf_lambda_window
        )
        self._prev_params: Optional[List[float]] = None
        self._fit_counter: int = 0
        self._scipy_checked: bool = False
        self._has_scipy: bool = False

    def _check_scipy(self) -> None:
        if self._scipy_checked:
            return
        self._scipy_checked = True
        try:
            import scipy.optimize  # noqa: F401
            import numpy  # noqa: F401

            self._has_scipy = True
        except ImportError:
            raise ImportError(
                "HopfDetectorStep requires scipy and numpy. "
                "Install with: pip install fracttalix[fast]"
            )

    def update(self, ctx: StepContext) -> None:
        if not ctx.config.enable_hopf_detector:
            return

        self._check_scipy()
        import numpy as np
        from scipy.optimize import curve_fit

        # Accumulate data
        val = ctx.current if hasattr(ctx, "current") else ctx.value
        if isinstance(val, (list, tuple)):
            val = val[0] if val else 0.0
        self._window.append(float(val))

        n = len(self._window)
        min_window = max(32, self.cfg.hopf_fit_window // 2)

        # Not enough data yet
        if n < min_window:
            self._write_empty(ctx)
            return

        # Only fit every N steps (performance)
        self._fit_counter += 1
        if self._fit_counter % self.cfg.hopf_fit_interval != 0:
            # Between fits, report last known values
            self._write_interpolated(ctx)
            return

        # --- FIT ---
        data = np.array(self._window, dtype=float)
        t = np.arange(n, dtype=float)

        # Determine ω constraint
        tau_gen = self.cfg.hopf_tau_gen
        if tau_gen is not None and tau_gen > 0:
            omega_predicted = math.pi / (2.0 * tau_gen)
        else:
            omega_predicted = _estimate_omega_from_fft(data)

        # Initialize parameters
        if self._prev_params is not None:
            B_init, A_init, lam_init, phi_init = self._prev_params
        else:
            B_init = float(np.mean(data[-max(1, n // 10) :]))
            A_init = float(np.max(np.abs(data - B_init)))
            if A_init < 1e-10:
                A_init = 1.0
            lam_init = _estimate_lambda_from_envelope(data)
            phi_init = 0.0

        # Fit with ω fixed at predicted value (3 free params + λ)
        def model(t_arr, B, A, lam, phi):
            return B + A * np.exp(-lam * t_arr) * np.cos(
                omega_predicted * t_arr + phi
            )

        try:
            popt, pcov = curve_fit(
                model,
                t,
                data,
                p0=[B_init, A_init, lam_init, phi_init],
                bounds=(
                    [-np.inf, -np.inf, 0.0, -2 * math.pi],
                    [np.inf, np.inf, 50.0, 2 * math.pi],
                ),
                maxfev=2000,
            )
            B_fit, A_fit, lam_fit, phi_fit = popt
            converged = True

            # Compute R²
            y_pred = model(t, *popt)
            ss_res = np.sum((data - y_pred) ** 2)
            ss_tot = np.sum((data - np.mean(data)) ** 2)
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

            # Warm-start for next fit
            self._prev_params = [B_fit, A_fit, lam_fit, phi_fit]

        except (RuntimeError, ValueError):
            # Fit failed — use last known values or defaults
            lam_fit = self._prev_params[2] if self._prev_params else 0.1
            B_fit = self._prev_params[0] if self._prev_params else 0.0
            A_fit = self._prev_params[1] if self._prev_params else 1.0
            phi_fit = self._prev_params[3] if self._prev_params else 0.0
            r_squared = 0.0
            converged = False

        # Track λ
        self._lambda_history.append(lam_fit)

        # Compute dλ/dt and time-to-bifurcation
        lam_rate, time_to_bif, confidence = self._compute_lambda_trend()

        # Scope status
        scope_status = self._compute_scope(r_squared, lam_fit, converged)

        # Alert logic
        alert, alert_type = self._compute_alert(
            lam_fit, time_to_bif, scope_status
        )

        # Implied τ_gen from fitted ω
        tau_gen_implied = (
            math.pi / (2.0 * omega_predicted)
            if omega_predicted > 1e-10
            else None
        )

        # Write to scratch
        ctx.scratch["hopf_lambda"] = lam_fit
        ctx.scratch["hopf_lambda_rate"] = lam_rate
        ctx.scratch["hopf_time_to_transition"] = time_to_bif
        ctx.scratch["hopf_confidence"] = confidence
        ctx.scratch["hopf_scope_status"] = scope_status
        ctx.scratch["hopf_r_squared"] = r_squared
        ctx.scratch["hopf_omega"] = omega_predicted
        ctx.scratch["hopf_tau_gen_implied"] = tau_gen_implied
        ctx.scratch["hopf_alert"] = alert
        ctx.scratch["hopf_alert_type"] = alert_type
        ctx.scratch["hopf_fit_converged"] = converged
        ctx.scratch["hopf_amplitude"] = A_fit
        ctx.scratch["hopf_baseline"] = B_fit

    def _compute_lambda_trend(self):
        """Compute dλ/dt and time-to-bifurcation from λ history."""
        history = list(self._lambda_history)
        if len(history) < 3:
            return 0.0, None, "LOW"

        import numpy as np

        lams = np.array(history, dtype=float)
        t = np.arange(len(lams), dtype=float)

        # OLS for dλ/dt
        t_mean = np.mean(t)
        lam_mean = np.mean(lams)
        numer = np.sum((t - t_mean) * (lams - lam_mean))
        denom = np.sum((t - t_mean) ** 2)
        if abs(denom) < 1e-12:
            return 0.0, None, "LOW"

        lam_rate = float(numer / denom)

        # Time-to-bifurcation (only when λ > 0 and declining)
        current_lam = lams[-1]
        time_to_bif = None
        if lam_rate < -1e-8 and current_lam > 0:
            time_to_bif = current_lam / abs(lam_rate)
            # Scale by fit_interval (we fit every N steps)
            time_to_bif *= self.cfg.hopf_fit_interval

        # Confidence from coefficient of variation of rate
        if len(lams) >= 4:
            diffs = np.diff(lams)
            rate_std = float(np.std(diffs))
            rate_mean = abs(float(np.mean(diffs)))
            if rate_mean > 1e-10:
                cv = rate_std / rate_mean
                if cv < 0.3:
                    confidence = "HIGH"
                elif cv < 0.7:
                    confidence = "MEDIUM"
                else:
                    confidence = "LOW"
            else:
                confidence = "LOW"
        else:
            confidence = "LOW"

        return lam_rate, time_to_bif, confidence

    def _compute_scope(self, r_squared, lam_fit, converged):
        """Determine whether the FRM model applies to current data."""
        if not converged:
            return "OUT_OF_SCOPE"
        if r_squared < self.cfg.hopf_r_squared_min:
            return "OUT_OF_SCOPE"
        if r_squared < 0.7:
            return "BOUNDARY"
        return "IN_SCOPE"

    def _compute_alert(self, lam, time_to_bif, scope_status):
        """Determine alert status from λ and scope."""
        if scope_status == "OUT_OF_SCOPE":
            return False, None

        # Check for critical slowing (λ below warning threshold)
        if lam < self.cfg.hopf_lambda_warning:
            if time_to_bif is not None and time_to_bif < self.cfg.hopf_t_decision:
                return True, "TRANSITION_APPROACHING"
            return True, "CRITICAL_SLOWING"

        # Check time-to-transition even if λ isn't below warning yet
        if (
            time_to_bif is not None
            and time_to_bif < 2.0 * self.cfg.hopf_t_decision
        ):
            return True, "TRANSITION_APPROACHING"

        return False, None

    def _write_empty(self, ctx: StepContext) -> None:
        """Write placeholder values when insufficient data."""
        ctx.scratch["hopf_lambda"] = None
        ctx.scratch["hopf_lambda_rate"] = None
        ctx.scratch["hopf_time_to_transition"] = None
        ctx.scratch["hopf_confidence"] = "LOW"
        ctx.scratch["hopf_scope_status"] = "INSUFFICIENT_DATA"
        ctx.scratch["hopf_r_squared"] = None
        ctx.scratch["hopf_omega"] = None
        ctx.scratch["hopf_tau_gen_implied"] = None
        ctx.scratch["hopf_alert"] = False
        ctx.scratch["hopf_alert_type"] = None
        ctx.scratch["hopf_fit_converged"] = False
        ctx.scratch["hopf_amplitude"] = None
        ctx.scratch["hopf_baseline"] = None

    def _write_interpolated(self, ctx: StepContext) -> None:
        """Write last-known values between fits."""
        if not self._lambda_history:
            self._write_empty(ctx)
            return

        lam_rate, time_to_bif, confidence = self._compute_lambda_trend()
        lam = self._lambda_history[-1]
        scope = "IN_SCOPE"  # assume last scope still valid between fits
        alert, alert_type = self._compute_alert(lam, time_to_bif, scope)

        ctx.scratch["hopf_lambda"] = lam
        ctx.scratch["hopf_lambda_rate"] = lam_rate
        ctx.scratch["hopf_time_to_transition"] = time_to_bif
        ctx.scratch["hopf_confidence"] = confidence
        ctx.scratch["hopf_scope_status"] = scope
        ctx.scratch["hopf_r_squared"] = None  # not computed this step
        ctx.scratch["hopf_omega"] = None
        ctx.scratch["hopf_tau_gen_implied"] = None
        ctx.scratch["hopf_alert"] = alert
        ctx.scratch["hopf_alert_type"] = alert_type
        ctx.scratch["hopf_fit_converged"] = False
        ctx.scratch["hopf_amplitude"] = None
        ctx.scratch["hopf_baseline"] = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "window": list(self._window),
            "lambda_history": list(self._lambda_history),
            "prev_params": self._prev_params,
            "fit_counter": self._fit_counter,
        }

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._window = deque(
            sd.get("window", []), maxlen=self.cfg.hopf_fit_window
        )
        self._lambda_history = deque(
            sd.get("lambda_history", []), maxlen=self.cfg.hopf_lambda_window
        )
        self._prev_params = sd.get("prev_params", None)
        self._fit_counter = sd.get("fit_counter", 0)
