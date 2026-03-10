# benchmark/ablation.py
# Ablation study: quantifies F1/VUS-PR contribution of each step group.
#
# Methodology: disable each step group in turn, measure F1 delta vs full pipeline.
# Step groups: foundation, temporal, frequency, complexity, channels, physics.
#
# Addresses hostile reviewer critique "are 37 steps justified?"

from typing import Any, Dict, List, Optional

from benchmark.archetypes import generate
from benchmark.metrics import _compute_f1_with_tolerance, _pr_auc, _vus_pr

# ---------------------------------------------------------------------------
# Step group definitions (step indices in the 37-step pipeline, 0-indexed)
# ---------------------------------------------------------------------------

# Steps 1-7 (indices 0-6): foundation
# NOTE: CoreEWMAStep (index 0) registers the "scalar" window bank that many
# downstream steps depend on (e.g. OscDampStep reads ctx.bank.get("scalar")).
# Disabling it crashes the pipeline. We exclude index 0 from "no_foundation" —
# it is architectural infrastructure, not merely a scoring step.
FOUNDATION_STEPS = list(range(1, 7))   # indices 1-6 (steps 2-7)
# Steps 8-11 (indices 7-10): temporal
TEMPORAL_STEPS = list(range(7, 11))
# Steps 12-15 (indices 11-15): frequency
FREQUENCY_STEPS = list(range(11, 15))
# Steps 16-20 (indices 15-20): complexity
COMPLEXITY_STEPS = list(range(15, 20))
# Steps 21-25 (indices 20-25): channels
CHANNELS_STEPS = list(range(20, 25))
# Steps 26-36 (indices 25-36): physics (AlertReasonsStep at 36 is always kept)
PHYSICS_STEPS = list(range(25, 36))

STEP_GROUPS: Dict[str, List[int]] = {
    "no_foundation": FOUNDATION_STEPS,
    "no_temporal": TEMPORAL_STEPS,
    "no_frequency": FREQUENCY_STEPS,
    "no_complexity": COMPLEXITY_STEPS,
    "no_channels": CHANNELS_STEPS,
    "no_physics": PHYSICS_STEPS,
}

STEP_GROUP_NAMES: Dict[str, str] = {
    "no_foundation": "Foundation (steps 1-7)",
    "no_temporal": "Temporal (steps 8-11)",
    "no_frequency": "Frequency (steps 12-15)",
    "no_complexity": "Complexity (steps 16-20)",
    "no_channels": "Channels (steps 21-25)",
    "no_physics": "Physics (steps 26-36)",
}


# ---------------------------------------------------------------------------
# Detector runner with optional step indices disabled
# ---------------------------------------------------------------------------

class _NopStep:
    """Replacement step that does nothing."""

    def update(self, ctx: Any) -> None:
        pass

    def reset(self) -> None:
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state(self, state: Dict[str, Any]) -> None:
        pass


def _run_with_disabled_steps(
    data: List[float],
    labels: List[int],
    disabled_indices: Optional[List[int]] = None,
    config: Optional[Any] = None,
) -> Dict[str, float]:
    """Run Sentinel with specific pipeline step indices replaced by no-ops.

    Parameters
    ----------
    data:
        Input time series.
    labels:
        Ground-truth anomaly labels.
    disabled_indices:
        0-based indices of pipeline steps to disable. None means full pipeline.
    config:
        Optional SentinelConfig.

    Returns
    -------
    dict with f1, auprc, vus_pr.
    """
    from fracttalix.config import SentinelConfig
    from fracttalix.detector import SentinelDetector
    from fracttalix.steps import _build_default_pipeline

    if config is None:
        config = SentinelConfig()

    pipeline = _build_default_pipeline(config)

    if disabled_indices:
        nop = _NopStep()
        for idx in disabled_indices:
            if 0 <= idx < len(pipeline):
                pipeline[idx] = nop  # type: ignore[assignment]

    det = SentinelDetector(config=config, steps=pipeline)
    results = [det.update_and_check(v) for v in data]

    scores = [float(r.get("anomaly_score", 0.0)) for r in results]
    preds = [bool(r.get("alert", False)) for r in results]

    _prec, _rec, f1 = _compute_f1_with_tolerance(preds, labels, tolerance=5)
    auprc = _pr_auc(scores, labels, tolerance=5)
    vus = _vus_pr(scores, labels)
    return {"f1": f1, "auprc": auprc, "vus_pr": vus}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ablation_study(
    archetype: str = "point",
    n: int = 1000,
    seed: int = 42,
    config: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Measure the contribution of each step group by disabling it.

    Parameters
    ----------
    archetype:
        Benchmark archetype (default "point").
    n:
        Dataset size (default 1000).
    seed:
        RNG seed (default 42).
    config:
        Optional SentinelConfig.

    Returns
    -------
    List of dicts, one per configuration:
        {
            "group": str,
            "steps_disabled": List[str],
            "f1": float,
            "vus_pr": float,
            "f1_delta": float,   # negative = performance drop from disabling
            "vus_pr_delta": float,
        }
    """
    data, labels = generate(archetype, n=n, seed=seed)

    # --- Baseline: full pipeline ---
    baseline = _run_with_disabled_steps(data, labels, disabled_indices=None, config=config)
    baseline_f1 = baseline["f1"]
    baseline_vus = baseline["vus_pr"]

    rows: List[Dict[str, Any]] = [
        {
            "group": "all_steps",
            "steps_disabled": [],
            "f1": baseline_f1,
            "vus_pr": baseline_vus,
            "f1_delta": 0.0,
            "vus_pr_delta": 0.0,
        }
    ]

    # --- Ablation per group ---
    for group_key, step_indices in STEP_GROUPS.items():
        result = _run_with_disabled_steps(
            data, labels, disabled_indices=step_indices, config=config
        )
        f1_delta = result["f1"] - baseline_f1
        vus_delta = result["vus_pr"] - baseline_vus

        # Build human-readable list of disabled step class names
        step_names = _get_step_names_for_indices(step_indices, config)

        rows.append(
            {
                "group": group_key,
                "steps_disabled": step_names,
                "f1": result["f1"],
                "vus_pr": result["vus_pr"],
                "f1_delta": f1_delta,
                "vus_pr_delta": vus_delta,
            }
        )

    print_ablation_table(rows, archetype=archetype)
    return rows


def _get_step_names_for_indices(
    indices: List[int],
    config: Optional[Any] = None,
) -> List[str]:
    """Return class names of pipeline steps at the given indices."""
    try:
        from fracttalix.config import SentinelConfig
        from fracttalix.steps import _build_default_pipeline

        if config is None:
            config = SentinelConfig()
        pipeline = _build_default_pipeline(config)
        names = []
        for idx in indices:
            if 0 <= idx < len(pipeline):
                names.append(type(pipeline[idx]).__name__)
        return names
    except Exception:
        return [f"step_{i}" for i in indices]


def print_ablation_table(
    results: List[Dict[str, Any]],
    archetype: str = "",
) -> None:
    """Print a formatted table showing each group's contribution."""
    title = (
        f"Ablation Study — archetype: {archetype}"
        if archetype
        else "Ablation Study"
    )
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"{title:^72}")
    print(sep)
    header = (
        f"{'Group':<22} {'F1':>7} {'VUS-PR':>8} "
        f"{'F1 Δ':>8} {'VUS Δ':>8}  Steps disabled"
    )
    print(header)
    print(sep)

    for row in results:
        group = row["group"]
        f1 = row["f1"]
        vus = row["vus_pr"]
        f1_d = row["f1_delta"]
        vus_d = row["vus_pr_delta"]
        disabled = row["steps_disabled"]

        # Abbreviate long lists
        if len(disabled) > 3:
            disabled_str = ", ".join(disabled[:3]) + f", +{len(disabled)-3} more"
        else:
            disabled_str = ", ".join(disabled) if disabled else "(none)"

        f1_d_str = f"{f1_d:+.3f}" if group != "all_steps" else "—"
        vus_d_str = f"{vus_d:+.3f}" if group != "all_steps" else "—"

        print(
            f"{group:<22} {f1:>7.3f} {vus:>8.3f} "
            f"{f1_d_str:>8} {vus_d_str:>8}  {disabled_str}"
        )
    print(f"{sep}\n")
