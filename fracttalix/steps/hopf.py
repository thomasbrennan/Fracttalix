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
    """Estimate dominant frequency from FFT with parabolic interpolation.

    Sub-bin resolution via Jacobsen/Quinn-style parabolic interpolation
    around the spectral peak gives ~10x better frequency accuracy than
    raw bin index alone. This matters for short windows where bin width
    is large relative to the true frequency.
    """
    import numpy as np

    n = len(data)
    centered = data - np.mean(data)
    # Apply Hann window to reduce spectral leakage
    window = np.hanning(n)
    windowed = centered * window
    fft_vals = np.abs(np.fft.rfft(windowed))

    if len(fft_vals) <= 2:
        return 0.1  # fallback

    # Find peak in magnitudes (skip DC)
    peak_idx = np.argmax(fft_vals[1:]) + 1

    # Parabolic interpolation for sub-bin accuracy
    if 1 < peak_idx < len(fft_vals) - 1:
        alpha = fft_vals[peak_idx - 1]
        beta = fft_vals[peak_idx]
        gamma = fft_vals[peak_idx + 1]
        denom = alpha - 2.0 * beta + gamma
        if abs(denom) > 1e-12:
            delta = 0.5 * (alpha - gamma) / denom
            refined_idx = peak_idx + delta
        else:
            refined_idx = float(peak_idx)
    else:
        refined_idx = float(peak_idx)

    omega = 2.0 * math.pi * refined_idx / n
    return max(omega, 0.01)


def _estimate_lambda_from_envelope(data):
    """Estimate decay rate from analytic signal envelope (Hilbert transform).

    Uses the Hilbert transform to extract the instantaneous amplitude
    envelope, then fits an exponential decay via log-linear regression.
    Much more robust than naive peak-picking, especially with noise.
    """
    import numpy as np

    n = len(data)
    if n < 8:
        return 0.1

    centered = data - np.mean(data)

    # Hilbert transform for analytic signal envelope
    try:
        from scipy.signal import hilbert
        analytic = hilbert(centered)
        envelope = np.abs(analytic)
    except ImportError:
        # Fallback: simple absolute value smoothed
        envelope = np.abs(centered)

    # Smooth envelope to suppress noise (moving average, width ~1/8 of window)
    smooth_width = max(3, n // 8)
    if smooth_width % 2 == 0:
        smooth_width += 1
    kernel = np.ones(smooth_width) / smooth_width
    envelope_smooth = np.convolve(envelope, kernel, mode="same")

    # Log-linear regression on smoothed envelope: log(A(t)) = -λt + c
    # Skip edges (convolution artifacts) and zero/near-zero values
    margin = smooth_width // 2
    t_vals = np.arange(margin, n - margin, dtype=float)
    env_vals = envelope_smooth[margin : n - margin]

    # Require envelope above noise floor (10% of max)
    noise_floor = 0.1 * np.max(env_vals) if np.max(env_vals) > 1e-12 else 1e-12
    mask = env_vals > noise_floor
    if mask.sum() < 4:
        return 0.1

    log_env = np.log(env_vals[mask])
    t_fit = t_vals[mask]

    # OLS: log(env) = slope * t + intercept
    t_mean = np.mean(t_fit)
    log_mean = np.mean(log_env)
    numer = np.sum((t_fit - t_mean) * (log_env - log_mean))
    denom = np.sum((t_fit - t_mean) ** 2)
    if abs(denom) < 1e-12:
        return 0.1
    slope = float(numer / denom)
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
            import numpy  # noqa: F401
            import scipy.optimize  # noqa: F401

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

        bounds = (
            [-np.inf, -np.inf, 0.0, -2 * math.pi],
            [np.inf, np.inf, 50.0, 2 * math.pi],
        )

        # Multi-start fitting: try primary initialization + 2 alternative
        # starts to avoid local minima. Keep the best fit by R².
        starts = [(B_init, A_init, lam_init, phi_init)]
        # Alternative: opposite amplitude sign, different phase
        starts.append((B_init, -A_init, lam_init * 0.5, math.pi / 2))
        # Alternative: zero damping start (for near-limit-cycle cases)
        starts.append((B_init, A_init, 0.01, -math.pi / 4))

        best_popt = None
        best_r2 = -np.inf
        any_converged = False

        ss_tot = np.sum((data - np.mean(data)) ** 2)
        if ss_tot < 1e-12:
            ss_tot = 1.0  # constant data edge case

        for p0 in starts:
            try:
                popt, _ = curve_fit(
                    model, t, data, p0=list(p0),
                    bounds=bounds, maxfev=2000,
                )
                y_pred = model(t, *popt)
                ss_res = np.sum((data - y_pred) ** 2)
                r2 = 1.0 - ss_res / ss_tot
                if r2 > best_r2:
                    best_r2 = r2
                    best_popt = popt
                    any_converged = True
            except (RuntimeError, ValueError):
                continue

        if any_converged:
            B_fit, A_fit, lam_fit, phi_fit = best_popt
            r_squared = best_r2
            converged = True
            self._prev_params = [B_fit, A_fit, lam_fit, phi_fit]
        else:
            lam_fit = self._prev_params[2] if self._prev_params else 0.1
            B_fit = self._prev_params[0] if self._prev_params else 0.0
            A_fit = self._prev_params[1] if self._prev_params else 1.0
            phi_fit = self._prev_params[3] if self._prev_params else 0.0
            r_squared = 0.0
            converged = False

        # Track λ (guard against NaN/Inf from failed fits)
        if math.isfinite(lam_fit):
            self._lambda_history.append(lam_fit)

        # Compute dλ/dt and time-to-bifurcation
        lam_rate, time_to_bif, confidence = self._compute_lambda_trend()

        # Scope status
        scope_status = self._compute_scope(r_squared, lam_fit, converged)

        # Alert logic
        alert, alert_type = self._compute_alert(
            lam_fit, lam_rate, time_to_bif, scope_status
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
        """Compute dλ/dt and time-to-bifurcation from λ history.

        Uses median-based outlier rejection before OLS regression.
        Lambda values that jump due to fitting artifacts (local minima,
        noise bursts) should not corrupt the trend estimate.
        """
        history = list(self._lambda_history)
        if len(history) < 3:
            return 0.0, None, "LOW"

        import numpy as np

        lams = np.array(history, dtype=float)
        t = np.arange(len(lams), dtype=float)

        # Outlier rejection: remove λ values > 3 MAD from median
        # This protects the trend from fitting artifacts
        median_lam = np.median(lams)
        mad = np.median(np.abs(lams - median_lam))
        if mad > 1e-12:
            threshold = 3.0 * mad / 0.6745  # MAD-based robust std
            inlier_mask = np.abs(lams - median_lam) < threshold
            if inlier_mask.sum() >= 3:
                lams_clean = lams[inlier_mask]
                t_clean = t[inlier_mask]
            else:
                lams_clean = lams
                t_clean = t
        else:
            lams_clean = lams
            t_clean = t

        # OLS for dλ/dt on cleaned data
        t_mean = np.mean(t_clean)
        lam_mean = np.mean(lams_clean)
        numer = np.sum((t_clean - t_mean) * (lams_clean - lam_mean))
        denom = np.sum((t_clean - t_mean) ** 2)
        if abs(denom) < 1e-12:
            return 0.0, None, "LOW"

        lam_rate = float(numer / denom)

        # Time-to-bifurcation (only when λ > 0 and declining)
        current_lam = lams[-1]
        time_to_bif = None
        if lam_rate < -1e-8 and current_lam > 0:
            time_to_bif = current_lam / abs(lam_rate)
            time_to_bif *= self.cfg.hopf_fit_interval

        # Confidence: uses R² of the trend line itself
        # High R² = consistent decline, Low R² = noisy/uncertain
        if len(lams_clean) >= 4:
            predicted = lam_mean + lam_rate * (t_clean - t_mean)
            ss_res = np.sum((lams_clean - predicted) ** 2)
            ss_tot = np.sum((lams_clean - lam_mean) ** 2)
            if ss_tot > 1e-12:
                trend_r2 = 1.0 - ss_res / ss_tot
                if trend_r2 > 0.6:
                    confidence = "HIGH"
                elif trend_r2 > 0.3:
                    confidence = "MEDIUM"
                else:
                    confidence = "LOW"
            else:
                confidence = "LOW"
        else:
            confidence = "LOW"

        return lam_rate, time_to_bif, confidence

    def _compute_scope(self, r_squared, lam_fit, converged):
        """Determine whether the FRM model applies to current data.

        Key distinction: the FRM applies to DAMPED oscillations (μ < 0).
        A sustained oscillation (constant amplitude, λ ≈ 0) is a limit
        cycle (μ ≥ 0) — out of scope for bifurcation detection. The
        detector must not fire CRITICAL_SLOWING on limit cycles.

        Detection: if λ is very small AND the amplitude envelope shows
        no decay across the window, the oscillation is sustained, not
        approaching bifurcation.
        """
        if not converged:
            return "OUT_OF_SCOPE"
        if r_squared < self.cfg.hopf_r_squared_min:
            return "OUT_OF_SCOPE"

        # Check for limit cycle: λ ≈ 0 with good fit means sustained
        # oscillation, not a damped system approaching bifurcation.
        # The test: exp(-λ * window) ≈ 1 means negligible decay.
        # Threshold 0.5 → less than ~39% amplitude decay across window.
        # Noise in sustained oscillations creates apparent small damping,
        # so the threshold must be generous enough to catch these.
        window_len = len(self._window)
        if window_len > 0 and lam_fit * window_len < 0.5:
            # Less than 39% amplitude decay across the window.
            # This is a sustained oscillation (limit cycle), not a
            # damped system losing its damping.
            return "LIMIT_CYCLE"

        if r_squared < 0.7:
            return "BOUNDARY"
        return "IN_SCOPE"

    def _compute_alert(self, lam, lam_rate, time_to_bif, scope_status):
        """Determine alert status from λ, its trend, and scope.

        Key insight: λ < threshold alone is not sufficient for an alert.
        A sustained oscillation (limit cycle) naturally has λ ≈ 0 with
        no declining trend — that's normal, not a warning sign.

        CRITICAL_SLOWING requires λ to be both small AND declining
        (lam_rate < 0). A stable small λ is a limit cycle.
        TRANSITION_APPROACHING uses time_to_bif which already requires
        declining λ (computed from negative lam_rate).
        """
        if scope_status in ("OUT_OF_SCOPE", "LIMIT_CYCLE"):
            return False, None

        # All alerts require λ to be actively declining (lam_rate < 0).
        # A sustained oscillation has λ ≈ 0 with stable/noisy rate —
        # not a system approaching bifurcation.
        if lam_rate >= -1e-3:
            # λ is not meaningfully declining — no alert
            return False, None

        # Check for critical slowing: λ below warning AND declining
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
        alert, alert_type = self._compute_alert(lam, lam_rate, time_to_bif, scope)

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
