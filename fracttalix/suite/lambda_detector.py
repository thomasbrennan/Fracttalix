# fracttalix/suite/lambda_detector.py
# LambdaDetector — FRM-derived bifurcation proximity via parametric λ tracking.
#
# What makes this unique:
#   - Fits the FRM form f(t) = B + A·exp(-λt)·cos(ωt + φ) to streaming data
#   - Tracks λ over time; when λ → 0 the system approaches Hopf bifurcation
#   - Provides time-to-transition estimate: Δt = λ / |dλ/dt|
#   - Uses ω = π/(2·τ_gen) constraint from FRM quarter-wave theorem
#
# No other detection system has this. Generic EWS (Scheffer et al.) watches
# statistical shadows (variance, AC1). This watches the dynamics directly.
#
# Requires: scipy (curve_fit, hilbert)

import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from fracttalix.suite.base import BaseDetector, _mean, _std, _variance


class LambdaDetector(BaseDetector):
    """Track the FRM decay rate λ toward zero (Hopf bifurcation proximity).

    When λ → 0, the system loses its damping and approaches a critical
    transition. This detector fits the FRM parametric form to a sliding
    window, estimates λ, tracks dλ/dt, and provides:

    - Time-to-transition estimate (no other detector can do this)
    - Scope awareness via R² (knows when FRM doesn't apply)
    - LIMIT_CYCLE detection (distinguishes sustained from damped oscillation)

    Parameters
    ----------
    tau_gen : float
        Generation timescale. ω = π/(2·τ_gen). If 0, estimated from FFT.
    fit_window : int
        Number of observations in the fitting window.
    fit_interval : int
        Fit every N steps (performance optimization).
    lambda_window : int
        Rolling window for λ history (trend estimation).
    lambda_warning : float
        λ threshold below which CRITICAL_SLOWING fires.
    r_squared_min : float
        Minimum R² for the FRM form to be considered in scope.
    """

    def __init__(
        self,
        tau_gen: float = 0.0,
        fit_window: int = 128,
        fit_interval: int = 4,
        lambda_window: int = 20,
        lambda_warning: float = 0.05,
        r_squared_min: float = 0.5,
        warmup: int = 0,
    ):
        # Warmup handled internally by fit_window fill
        super().__init__(
            name="Lambda",
            warmup=warmup,
            window_size=fit_window,
        )
        self._tau_gen = tau_gen
        self._fit_interval = fit_interval
        self._lambda_window = lambda_window
        self._lambda_warning = lambda_warning
        self._r_squared_min = r_squared_min
        self._alert_threshold = 0.5

        # Internal state
        self._lambda_history: deque = deque(maxlen=lambda_window)
        self._prev_params: Optional[List[float]] = None
        self._fit_counter = 0
        self._last_r_squared = 0.0
        self._last_lambda = None
        self._last_lam_rate = 0.0
        self._last_time_to_bif = None
        self._last_scope = "INSUFFICIENT_DATA"
        self._scipy_available: Optional[bool] = None

    def _check_scipy(self) -> bool:
        if self._scipy_available is None:
            try:
                import numpy as np  # noqa: F401
                from scipy.optimize import curve_fit  # noqa: F401
                self._scipy_available = True
            except ImportError:
                self._scipy_available = False
        return self._scipy_available

    def _check_scope(self, window: List[float]) -> bool:
        if not self._check_scipy():
            return False
        min_window = max(32, self._window_size // 2)
        if len(window) < min_window:
            return False
        # Scope is determined during _compute via R²
        # We allow _compute to run and return OUT_OF_SCOPE score internally
        return True

    def _compute(self, window: List[float]) -> Tuple[float, str]:
        import numpy as np
        from scipy.optimize import curve_fit

        self._fit_counter += 1

        # Between fits, use last known values
        if self._fit_counter % self._fit_interval != 0:
            return self._score_from_state()

        data = np.array(window, dtype=float)
        n = len(data)
        t = np.arange(n, dtype=float)

        # Determine ω
        if self._tau_gen and self._tau_gen > 0:
            omega = math.pi / (2.0 * self._tau_gen)
        else:
            omega = self._estimate_omega(data)

        # Initialize parameters
        if self._prev_params is not None:
            B_init, A_init, lam_init, phi_init = self._prev_params
        else:
            B_init = float(np.mean(data[-max(1, n // 10):]))
            A_init = float(np.max(np.abs(data - B_init)))
            if A_init < 1e-10:
                A_init = 1.0
            lam_init = self._estimate_lambda_envelope(data)
            phi_init = 0.0

        # FRM model with ω fixed
        def model(t_arr, B, A, lam, phi):
            return B + A * np.exp(-lam * t_arr) * np.cos(omega * t_arr + phi)

        bounds = (
            [-np.inf, -np.inf, 0.0, -2 * math.pi],
            [np.inf, np.inf, 50.0, 2 * math.pi],
        )

        # Multi-start fitting
        starts = [
            (B_init, A_init, lam_init, phi_init),
            (B_init, -A_init, lam_init * 0.5, math.pi / 2),
            (B_init, A_init, 0.01, -math.pi / 4),
        ]

        best_popt = None
        best_r2 = -np.inf
        ss_tot = np.sum((data - np.mean(data)) ** 2)
        if ss_tot < 1e-12:
            ss_tot = 1.0

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
            except (RuntimeError, ValueError):
                continue

        if best_popt is None:
            self._last_r_squared = 0.0
            self._last_scope = "OUT_OF_SCOPE"
            return 0.0, "fit failed"

        B_fit, A_fit, lam_fit, phi_fit = best_popt
        self._last_r_squared = best_r2
        self._prev_params = [B_fit, A_fit, lam_fit, phi_fit]
        self._last_lambda = lam_fit

        # Track λ
        self._lambda_history.append(lam_fit)

        # Compute scope
        self._last_scope = self._compute_scope(best_r2, lam_fit, n)

        # Compute trend
        self._last_lam_rate, self._last_time_to_bif = self._compute_trend()

        return self._score_from_state()

    def _compute_scope(self, r2: float, lam: float, window_len: int) -> str:
        if r2 < self._r_squared_min:
            return "OUT_OF_SCOPE"
        if window_len > 0 and lam * window_len < 0.5:
            return "LIMIT_CYCLE"
        if r2 < 0.7:
            return "BOUNDARY"
        return "IN_SCOPE"

    def _compute_trend(self) -> Tuple[float, Optional[float]]:
        import numpy as np

        history = list(self._lambda_history)
        if len(history) < 3:
            return 0.0, None

        lams = np.array(history, dtype=float)
        t = np.arange(len(lams), dtype=float)

        # MAD outlier rejection
        median_lam = np.median(lams)
        mad = np.median(np.abs(lams - median_lam))
        if mad > 1e-12:
            threshold = 3.0 * mad / 0.6745
            mask = np.abs(lams - median_lam) < threshold
            if mask.sum() >= 3:
                lams = lams[mask]
                t = t[mask]

        t_mean = np.mean(t)
        lam_mean = np.mean(lams)
        numer = np.sum((t - t_mean) * (lams - lam_mean))
        denom = np.sum((t - t_mean) ** 2)
        if abs(denom) < 1e-12:
            return 0.0, None

        rate = float(numer / denom)

        # Time-to-bifurcation
        time_to_bif = None
        current = self._lambda_history[-1]
        if rate < -1e-8 and current > 0:
            time_to_bif = current / abs(rate) * self._fit_interval

        return rate, time_to_bif

    def _score_from_state(self) -> Tuple[float, str]:
        if self._last_scope in ("OUT_OF_SCOPE", "LIMIT_CYCLE"):
            msg = f"scope={self._last_scope}"
            if self._last_lambda is not None:
                msg += f" λ={self._last_lambda:.4f} R²={self._last_r_squared:.3f}"
            return 0.0, msg

        lam = self._last_lambda
        rate = self._last_lam_rate
        ttb = self._last_time_to_bif

        if lam is None:
            return 0.0, "no fit yet"

        # Score: how close to bifurcation?
        # λ < warning AND declining → high score
        if rate >= -1e-3:
            # Not declining — normal
            score = max(0.0, min(0.49, (self._lambda_warning - lam) / self._lambda_warning * 0.3))
            return score, f"λ={lam:.4f} rate={rate:.4f} stable"

        # λ is declining
        if lam < self._lambda_warning:
            if ttb is not None and ttb < 40.0:
                score = min(1.0, 0.7 + 0.3 * (40.0 - ttb) / 40.0)
                return score, f"TRANSITION λ={lam:.4f} rate={rate:.4f} Δt={ttb:.1f}"
            score = 0.6
            return score, f"CRITICAL_SLOWING λ={lam:.4f} rate={rate:.4f}"

        # λ above warning but declining
        score = max(0.0, min(0.49, 0.3 * abs(rate) / 0.01))
        return score, f"λ={lam:.4f} rate={rate:.4f} declining"

    def _estimate_omega(self, data) -> float:
        import numpy as np
        n = len(data)
        centered = data - np.mean(data)
        window = np.hanning(n)
        fft_vals = np.abs(np.fft.rfft(centered * window))
        if len(fft_vals) <= 2:
            return 0.1
        peak_idx = np.argmax(fft_vals[1:]) + 1
        if 1 < peak_idx < len(fft_vals) - 1:
            a, b, g = fft_vals[peak_idx - 1], fft_vals[peak_idx], fft_vals[peak_idx + 1]
            d = a - 2.0 * b + g
            if abs(d) > 1e-12:
                delta = 0.5 * (a - g) / d
                refined = peak_idx + delta
            else:
                refined = float(peak_idx)
        else:
            refined = float(peak_idx)
        return max(2.0 * math.pi * refined / n, 0.01)

    def _estimate_lambda_envelope(self, data) -> float:
        import numpy as np
        n = len(data)
        if n < 8:
            return 0.1
        centered = data - np.mean(data)
        try:
            from scipy.signal import hilbert
            envelope = np.abs(hilbert(centered))
        except ImportError:
            envelope = np.abs(centered)
        sw = max(3, n // 8)
        if sw % 2 == 0:
            sw += 1
        kernel = np.ones(sw) / sw
        env_smooth = np.convolve(envelope, kernel, mode="same")
        margin = sw // 2
        t_vals = np.arange(margin, n - margin, dtype=float)
        env_vals = env_smooth[margin:n - margin]
        floor = 0.1 * np.max(env_vals) if np.max(env_vals) > 1e-12 else 1e-12
        mask = env_vals > floor
        if mask.sum() < 4:
            return 0.1
        log_env = np.log(env_vals[mask])
        t_fit = t_vals[mask]
        t_mean = np.mean(t_fit)
        log_mean = np.mean(log_env)
        numer = np.sum((t_fit - t_mean) * (log_env - log_mean))
        denom = np.sum((t_fit - t_mean) ** 2)
        if abs(denom) < 1e-12:
            return 0.1
        return max(-float(numer / denom), 0.0)

    @property
    def current_lambda(self) -> Optional[float]:
        return self._last_lambda

    @property
    def lambda_rate(self) -> float:
        return self._last_lam_rate

    @property
    def time_to_transition(self) -> Optional[float]:
        return self._last_time_to_bif

    @property
    def r_squared(self) -> float:
        return self._last_r_squared

    @property
    def scope_status(self) -> str:
        return self._last_scope

    def reset(self) -> None:
        super().reset()
        self._lambda_history.clear()
        self._prev_params = None
        self._fit_counter = 0
        self._last_r_squared = 0.0
        self._last_lambda = None
        self._last_lam_rate = 0.0
        self._last_time_to_bif = None
        self._last_scope = "INSUFFICIENT_DATA"

    def state_dict(self) -> Dict[str, Any]:
        sd = super().state_dict()
        sd.update({
            "lambda_history": list(self._lambda_history),
            "prev_params": self._prev_params,
            "fit_counter": self._fit_counter,
            "last_r_squared": self._last_r_squared,
            "last_lambda": self._last_lambda,
            "last_lam_rate": self._last_lam_rate,
        })
        return sd

    def load_state(self, sd: Dict[str, Any]) -> None:
        super().load_state(sd)
        self._lambda_history = deque(
            sd.get("lambda_history", []), maxlen=self._lambda_window
        )
        self._prev_params = sd.get("prev_params")
        self._fit_counter = sd.get("fit_counter", 0)
        self._last_r_squared = sd.get("last_r_squared", 0.0)
        self._last_lambda = sd.get("last_lambda")
        self._last_lam_rate = sd.get("last_lam_rate", 0.0)
