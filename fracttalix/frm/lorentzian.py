# fracttalix/frm/lorentzian.py
# Lorentzian spectral fit utility — shared by LambdaDetector and OmegaDetector.
#
# Physical basis:
#   Near a Hopf bifurcation the power spectrum of a noisy oscillator is
#   Lorentzian in the vicinity of the fundamental frequency:
#
#     S(f) = A / ((f - f0)^2 + gamma^2) + B
#
#   where  gamma = lambda / (2*pi)  is the Lorentzian half-width.
#   Inverting:  lambda = 2*pi*gamma.
#
#   This holds for the Stuart-Landau oscillator and other noisy limit-cycle
#   systems near bifurcation (not just the OU linearisation), making it
#   applicable to real nonlinear data.
#
# Phase-diffusion immunity:
#   Autocorrelation lag estimation is sensitive to phase noise: in a noisy
#   oscillator the instantaneous frequency scatters by D=sigma^2/(2A^2) per
#   step, causing >5% period scatter.  The Lorentzian CENTROID f0 is immune:
#   phase diffusion broadens the peak (increases gamma) but does not shift f0.
#   Using f0_fit from fit_lorentzian instead of autocorrelation lag eliminates
#   spurious Omega ALERT on stable limit cycles.
#
# Implemented for the FRM physics layer.

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def welch_psd(
    data: np.ndarray,
    seg_len: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD estimate with 50% overlap and Hann windowing.

    Returns (freqs, psd) in cycles-per-sample units (normalised to [0, 0.5]).
    The returned PSD is amplitude-scaled so that the peak height approximately
    equals the true spectral density (divided by the squared Hann norm).

    Parameters
    ----------
    data    : 1-D array of floats (already detrended if desired)
    seg_len : segment length (default N//2, min 16)
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if seg_len is None:
        seg_len = max(16, n // 2)
    seg_len = min(seg_len, n)
    step = max(1, seg_len // 2)  # 50% overlap

    hann = np.hanning(seg_len)
    hann_sq_sum = float(np.sum(hann ** 2))

    segments = []
    start = 0
    while start + seg_len <= n:
        seg = data[start : start + seg_len]
        seg_c = (seg - np.mean(seg)) * hann
        segments.append(np.abs(np.fft.rfft(seg_c)) ** 2)
        start += step

    if not segments:
        # Degenerate: window larger than data — single-segment fallback
        seg_c = (data - np.mean(data)) * np.hanning(n)
        psd = np.abs(np.fft.rfft(seg_c)) ** 2 / (float(np.sum(np.hanning(n) ** 2)) or 1.0)
        freqs = np.fft.rfftfreq(n)
        return freqs, psd

    psd = np.mean(np.array(segments), axis=0) / (hann_sq_sum or 1.0)
    freqs = np.fft.rfftfreq(seg_len)
    return freqs, psd


def _lorentzian_plus_floor(f, f0, gamma, A, B):
    """Lorentzian peak plus flat floor: S(f) = A/((f-f0)^2+gamma^2) + B."""
    return A / ((f - f0) ** 2 + gamma ** 2) + B


def fit_lorentzian(
    freqs: np.ndarray,
    psd: np.ndarray,
    f0_pred: Optional[float] = None,
    band_factor: float = 0.5,
) -> Tuple[float, float, float, bool]:
    """Fit a Lorentzian spectral peak to PSD near f0_pred.

    Model: S(f) = A / ((f - f0)^2 + gamma^2) + B
    where  gamma = lambda / (2*pi),  so  lambda_fit = 2*pi*gamma.

    Parameters
    ----------
    freqs      : frequency array from welch_psd (cycles/sample, excluding DC)
    psd        : PSD array from welch_psd
    f0_pred    : predicted peak frequency (cycles/sample), or None to find peak
    band_factor: fit only within ±band_factor of f0_pred (default ±50%)

    Returns
    -------
    f0_fit          : fitted peak frequency (cycles/sample)
    lambda_fit      : estimated damping rate (2*pi*gamma)
    r_squared       : goodness-of-fit ∈ [0, 1];  < 0.5 indicates poor fit
    fwhm_resolvable : True when FWHM (= 2*gamma) > 1 frequency bin
    """
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        # scipy unavailable: return peak frequency, zero lambda, zero quality
        peak_idx = int(np.argmax(psd))
        return (float(freqs[peak_idx]), 0.0, 0.0, False)

    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)

    # Strip DC bin if present
    start = 1 if (len(freqs) > 0 and freqs[0] < 1e-10) else 0
    freqs = freqs[start:]
    psd = psd[start:]

    if len(freqs) < 5:
        f_fallback = f0_pred if f0_pred is not None else 0.0
        return (f_fallback, 0.0, 0.0, False)

    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

    # Restrict to band around predicted frequency
    if f0_pred is not None and f0_pred > 0:
        f_lo = f0_pred * (1.0 - band_factor)
        f_hi = f0_pred * (1.0 + band_factor)
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        if mask.sum() >= 5:
            f_band = freqs[mask]
            p_band = psd[mask]
        else:
            # Band too narrow — use all frequencies
            f_band = freqs
            p_band = psd
    else:
        f_band = freqs
        p_band = psd

    if len(f_band) < 5:
        peak_idx = int(np.argmax(psd))
        return (float(freqs[peak_idx]), 0.0, 0.0, False)

    # Initial parameter estimates
    peak_idx = int(np.argmax(p_band))
    f0_init = float(f_band[peak_idx])
    p_peak = float(p_band[peak_idx])
    B_init = float(np.percentile(p_band, 10))
    # A = (peak - B) * gamma^2  → choose gamma = df initially
    gamma_init = max(df, (float(f_band[-1]) - float(f_band[0])) * 0.05)
    A_init = max((p_peak - B_init) * gamma_init ** 2, df ** 2 * p_peak * 1e-3)

    # Bounds: f0 within band, gamma ∈ [0.1*df, half band width], A ≥ 0, B ≥ 0
    f0_lo = float(f_band[0])
    f0_hi = float(f_band[-1])
    gamma_lo = df * 0.05
    gamma_hi = max((f0_hi - f0_lo) * 0.5, df * 2.0)
    A_lo = 0.0
    A_hi = p_peak * (gamma_hi ** 2) * 10.0
    B_lo = 0.0
    B_hi = float(np.max(p_band))

    try:
        popt, _ = curve_fit(
            _lorentzian_plus_floor,
            f_band,
            p_band,
            p0=[f0_init, gamma_init, A_init, B_init],
            bounds=(
                [f0_lo, gamma_lo, A_lo, B_lo],
                [f0_hi, gamma_hi, A_hi, B_hi],
            ),
            maxfev=3000,
        )
        f0_fit, gamma_fit, A_fit, B_fit = popt
    except Exception:
        # Fit failed — return peak position, poor quality
        return (f0_init, 2.0 * math.pi * gamma_init, 0.0, False)

    # R² over the fitted band
    p_pred = _lorentzian_plus_floor(f_band, *popt)
    ss_res = float(np.sum((p_band - p_pred) ** 2))
    ss_tot = float(np.sum((p_band - np.mean(p_band)) ** 2))
    r_squared = float(np.clip(1.0 - ss_res / (ss_tot + 1e-30), 0.0, 1.0))

    gamma_fit = abs(float(gamma_fit))
    fwhm = 2.0 * gamma_fit
    fwhm_resolvable = fwhm > df

    lambda_fit = 2.0 * math.pi * gamma_fit
    return (float(f0_fit), lambda_fit, r_squared, fwhm_resolvable)
