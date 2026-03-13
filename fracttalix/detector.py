# fracttalix/detector.py
# SentinelDetector class (with auto_tune, reset, save_state, load_state, plot_history)
# Detector_7_10 backward compat alias
# _legacy_kwargs_to_config() helper

import csv
import dataclasses
import json
import math
import warnings
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from fracttalix._compat import _MPL, plt
from fracttalix.config import SentinelConfig
from fracttalix.steps import _build_default_pipeline
from fracttalix.steps.base import DetectorStep
from fracttalix.types import SentinelResult
from fracttalix.window import StepContext, WindowBank

try:
    from fracttalix import __version__
except Exception:
    __version__ = "12.3.0"


def _legacy_kwargs_to_config(kw: dict) -> SentinelConfig:
    """Map v7.x flat kwargs to SentinelConfig (backward compat)."""
    mapping = {
        "alpha": "alpha",
        "dev_alpha": "dev_alpha",
        "multiplier": "multiplier",
        "warmup_periods": "warmup_periods",
        "regime_threshold": "regime_threshold",
        "regime_alpha_boost": "regime_alpha_boost",
        "multivariate": "multivariate",
        "n_channels": "n_channels",
        "cov_alpha": "cov_alpha",
        "rpi_window": "rpi_window",
        "rfi_window": "rfi_window",
        "rpi_threshold": "rpi_threshold",
        "rfi_threshold": "rfi_threshold",
        "pe_order": "pe_order",
        "pe_window": "pe_window",
        "pe_threshold": "pe_threshold",
        "ews_window": "ews_window",
        "ews_threshold": "ews_threshold",
        "sti_window": "sti_window",
        "tps_window": "tps_window",
        "osc_damp_window": "osc_damp_window",
        "osc_threshold": "osc_threshold",
        "cpd_window": "cpd_window",
        "cpd_threshold": "cpd_threshold",
        "ph_delta": "ph_delta",
        "ph_lambda": "ph_lambda",
        "var_cusum_k": "var_cusum_k",
        "var_cusum_h": "var_cusum_h",
        "seasonal_period": "seasonal_period",
        "quantile_threshold_mode": "quantile_threshold_mode",
        "aqb_window": "aqb_window",
        "aqb_q_low": "aqb_q_low",
        "aqb_q_high": "aqb_q_high",
        "history_maxlen": "history_maxlen",
        "csv_path": "csv_path",
    }
    mapped = {}
    for old, new in mapping.items():
        if old in kw:
            mapped[new] = kw[old]
    # v7.x compat: 'rsi_window' -> rpi_window (they shared same window)
    if "rsi_window" in kw and "rpi_window" not in mapped:
        mapped["rpi_window"] = kw["rsi_window"]
    return SentinelConfig(**mapped)


class SentinelDetector:
    """Streaming anomaly detector — v8.0 pipeline architecture.

    Usage::

        det = SentinelDetector()
        for value in stream:
            result = det.update_and_check(value)
            if result["alert"]:
                print(result["alert_reasons"])

    Backward-compatible with all v7.x kwargs.
    ``Detector_7_10`` is an alias for this class.
    """

    def __init__(self, config: Optional[SentinelConfig] = None,
                 *, steps: Optional[List[DetectorStep]] = None,
                 **legacy_kwargs):
        if config is None:
            if legacy_kwargs:
                config = _legacy_kwargs_to_config(legacy_kwargs)
            else:
                config = SentinelConfig.production()
        self.config = config
        self._bank = WindowBank()
        if steps is not None:
            self._pipeline = list(steps)
        else:
            self._pipeline = _build_default_pipeline(config)
        # Phase 1.3: _core_step_ref side channel removed.
        # CoreEWMAStep and RegimeStep are wired via RegimeBoostState at
        # construction time in _build_default_pipeline. No scratch key needed.

        self._n = 0
        self._history: deque = deque(maxlen=config.history_maxlen)
        self._csv_file = None
        self._csv_writer = None
        if config.csv_path:
            self._open_csv(config.csv_path)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update_and_check(self, value) -> SentinelResult:
        """Process one observation; return SentinelResult (dict subclass)."""
        ctx = StepContext(
            value=value,
            step=self._n,
            config=self.config,
            bank=self._bank,
            scratch={},
        )
        # Phase 1.3: no _core_step_ref injection needed — boost via RegimeBoostState
        for step in self._pipeline:
            step.update(ctx)

        result = SentinelResult(
            (k, v) for k, v in ctx.scratch.items()
            if not k.startswith("_")
        )
        result.setdefault("alert", False)
        result.setdefault("anomaly", False)
        result.setdefault("warmup", ctx.is_warmup)
        result.setdefault("z_score", 0.0)
        result.setdefault("anomaly_score", 0.0)
        result.setdefault("alert_reasons", [])
        # V9.0 defaults
        result.setdefault("frequency_bands", None)
        result.setdefault("structural_snapshot", None)
        result.setdefault("coupling_matrix", None)
        result.setdefault("channel_coherence", None)
        result.setdefault("degradation_sequence", None)
        result.setdefault("cascade_precursor_active", False)
        result.setdefault("band_anomalies", {})
        result.setdefault("channel_summary", "")
        result.setdefault("v9_active_alerts", [])
        # V10.0 defaults
        result.setdefault("maintenance_burden", 0.0)
        result.setdefault("tainter_regime", "UNKNOWN")
        result.setdefault("throughput", 0.0)
        result.setdefault("mean_pac", 0.0)
        result.setdefault("pac_degradation_rate", 0.0)
        result.setdefault("pre_cascade_pac", False)
        result.setdefault("diagnostic_window_steps", None)
        result.setdefault("diagnostic_window_steps_pessimistic", None)
        result.setdefault("diagnostic_window_steps_optimistic", None)
        result.setdefault("diagnostic_window_confidence", "NOT_APPLICABLE")
        result.setdefault("supercompensation_detected", False)
        result.setdefault("kuramoto_order", 0.0)
        result.setdefault("kuramoto_order_v10", 0.0)
        # Phase 3.4: phi_kappa_separation = Phi - kappa_bar
        # Positive: phase coherence exceeds coupling strength (intervention signature)
        # Negative: coupling strength exceeds phase coherence (organic degradation)
        phi = result.get("kuramoto_order", 0.0)
        kappa = result.get("mean_coupling_strength", 0.0)
        result["phi_kappa_separation"] = phi - kappa
        result.setdefault("reversed_sequence", False)
        result.setdefault("intervention_signature_score", 0.0)
        result.setdefault("sequence_type", "INSUFFICIENT_DATA")
        result.setdefault("coupling_rate", 0.0)
        result.setdefault("critical_coupling", 0.5)
        result.setdefault("critical_coupling_v10", 0.5)
        result.setdefault("maintenance_burden_v10", 0.0)
        result.setdefault("kuramoto_order_v10", 0.0)
        result.setdefault("phi_kappa_separation", 0.0)
        result.setdefault("mean_coupling_strength", 0.0)
        result["step"] = self._n
        result["value"] = value if not isinstance(value, (list, tuple)) else list(value)

        self._n += 1
        self._history.append(result)
        if self._csv_writer is not None:
            self._write_csv_row(result)
        return result

    async def aupdate(self, value) -> Dict[str, Any]:
        """Async wrapper for update_and_check."""
        return self.update_and_check(value)

    # ------------------------------------------------------------------
    # Fitting / tuning
    # ------------------------------------------------------------------

    def fit(self, data: Sequence) -> "SentinelDetector":
        """Warm up detector on unlabeled data (returns self)."""
        for v in data:
            self.update_and_check(v)
        return self

    @classmethod
    def auto_tune(cls, data: Sequence,
                  labeled_data: Optional[Sequence[Tuple[Any, bool]]] = None,
                  alphas: Optional[List[float]] = None,
                  multipliers: Optional[List[float]] = None) -> "SentinelDetector":
        """Grid-search alpha/multiplier to maximise F1 on labeled_data
        (or minimise false-positive rate if labels not provided).
        """
        if alphas is None:
            alphas = [0.05, 0.1, 0.2, 0.3]
        if multipliers is None:
            multipliers = [2.0, 2.5, 3.0, 3.5, 4.0]

        best_score = -1.0
        best_cfg: Optional[SentinelConfig] = None

        for a in alphas:
            for m in multipliers:
                cfg = SentinelConfig(alpha=a, dev_alpha=a, multiplier=m)
                det = cls(config=cfg)
                if labeled_data is not None:
                    results = []
                    labels = []
                    for v, lbl in labeled_data:
                        r = det.update_and_check(v)
                        results.append(r)
                        labels.append(lbl)
                    # Compute F1
                    tp = fp = fn = 0
                    for r, lbl in zip(results, labels):
                        pred = r["alert"]
                        if pred and lbl: tp += 1
                        elif pred and not lbl: fp += 1
                        elif not pred and lbl: fn += 1
                    prec = tp / (tp + fp + 1e-10)
                    rec = tp / (tp + fn + 1e-10)
                    f1 = 2 * prec * rec / (prec + rec + 1e-10)
                    score = f1
                else:
                    results = [det.update_and_check(v) for v in data]
                    fp_rate = sum(1 for r in results if r["alert"] and not r["warmup"]) / (len(results) + 1)
                    score = 1.0 - fp_rate

                if score > best_score:
                    best_score = score
                    best_cfg = cfg

        return cls(config=best_cfg or SentinelConfig())

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def save_state(self) -> str:
        """Serialize full detector state to JSON string."""
        # Import __version__ lazily to avoid circular imports
        try:
            from fracttalix import __version__ as _ver
        except Exception:
            _ver = "12.0.0"
        sd: Dict[str, Any] = {
            "version": _ver,
            "n": self._n,
            "config": dataclasses.asdict(self.config),
            "bank": self._bank.state_dict(),
            "steps": [],
        }
        for i, step in enumerate(self._pipeline):
            sd["steps"].append({
                "cls": type(step).__name__,
                "idx": i,
                "state": step.state_dict(),
            })
        return json.dumps(sd)

    def load_state(self, json_str: str) -> None:
        """Restore detector state from JSON string."""
        sd = json.loads(json_str)
        self._n = sd.get("n", 0)
        self._bank.load_state(sd.get("bank", {}))
        step_states = sd.get("steps", [])
        for ss in step_states:
            idx = ss.get("idx", -1)
            if 0 <= idx < len(self._pipeline):
                self._pipeline[idx].load_state(ss.get("state", {}))

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, soft: bool = False) -> None:
        """Reset detector state.

        Parameters
        ----------
        soft:
            If True, only reset accumulators but keep warmup data.
            If False (default), full reset to factory state.
        """
        self._n = 0
        self._history.clear()
        self._bank.reset()
        for step in self._pipeline:
            step.reset()

    # ------------------------------------------------------------------
    # CSV / history
    # ------------------------------------------------------------------

    def _open_csv(self, path: str) -> None:
        self._csv_file = open(path, "w", newline="")
        self._csv_writer = None  # created lazily on first row

    def _write_csv_row(self, result: Dict[str, Any]) -> None:
        if self._csv_writer is None:
            fieldnames = [k for k in result.keys() if k != "alert_reasons"]
            fieldnames.append("alert_reasons")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames,
                                               extrasaction="ignore")
            self._csv_writer.writeheader()
        row = dict(result)
        row["alert_reasons"] = "|".join(result.get("alert_reasons", []))
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def flush_csv(self) -> None:
        """Flush pending CSV writes."""
        if self._csv_file is not None:
            self._csv_file.flush()

    def close(self) -> None:
        """Close CSV file if open."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_history(self, title: str = "Sentinel v8 Dashboard",
                     show: bool = True) -> Optional[Any]:
        """4-panel dashboard: value+EWMA, anomaly_score, PE, z_score."""
        if not _MPL:
            warnings.warn("matplotlib not available")
            return None
        history = list(self._history)
        if not history:
            return None

        steps = [r["step"] for r in history]
        values = [r.get("value", 0) if not isinstance(r.get("value"), list) else r["value"][-1]
                  for r in history]
        ewmas = [r.get("ewma", 0) for r in history]
        scores = [r.get("anomaly_score", 0) for r in history]
        pes = [r.get("pe", 0.5) for r in history]
        zs = [r.get("z_score", 0) for r in history]
        alerts = [i for i, r in enumerate(history) if r.get("alert")]

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(title, fontsize=14, fontweight="bold")

        axes[0].plot(steps, values, color="steelblue", linewidth=0.8, label="Value")
        axes[0].plot(steps, ewmas, color="orange", linewidth=1.5, label="EWMA")
        for ai in alerts:
            axes[0].axvline(steps[ai], color="red", alpha=0.3, linewidth=0.5)
        axes[0].set_ylabel("Value / EWMA")
        axes[0].legend(fontsize=8)

        axes[1].plot(steps, scores, color="purple", linewidth=0.8)
        axes[1].axhline(1.0, color="red", linestyle="--", linewidth=0.8)
        axes[1].set_ylabel("Anomaly Score")

        axes[2].plot(steps, pes, color="teal", linewidth=0.8)
        axes[2].set_ylabel("Permutation Entropy")

        axes[3].plot(steps, zs, color="gray", linewidth=0.8)
        axes[3].axhline(self.config.multiplier, color="red", linestyle="--", linewidth=0.8)
        axes[3].axhline(-self.config.multiplier, color="red", linestyle="--", linewidth=0.8)
        axes[3].set_ylabel("Z-score")
        axes[3].set_xlabel("Step")

        plt.tight_layout()
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        try:
            from fracttalix import __version__ as _ver
        except Exception:
            _ver = "12.0.0"
        return (f"SentinelDetector(v{_ver}, n={self._n}, "
                f"alpha={self.config.alpha}, warmup={self.config.warmup_periods}, "
                f"three_channel=True)")

    def __len__(self) -> int:
        return self._n


# Backward-compat alias (ε)
Detector_7_10 = SentinelDetector
