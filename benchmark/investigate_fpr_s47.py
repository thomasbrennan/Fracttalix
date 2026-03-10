#!/usr/bin/env python3
"""
FALSE POSITIVE RATE INVESTIGATION — v12.1
Work Order S47 | Session: thomasbrennan/Fracttalix
"""
import sys

sys.path.insert(0, '/home/user/Fracttalix')

from benchmark.ablation import _NopStep
from benchmark.archetypes import ARCHETYPES, _randn_list, generate
from benchmark.metrics import evaluate
from fracttalix.config import SentinelConfig
from fracttalix.detector import SentinelDetector
from fracttalix.steps import _build_default_pipeline
from fracttalix.steps.base import DetectorStep

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def alert_rate_on_noise(config, n=1000, seed=99, disabled_indices=None):
    """Normal alert rate on pure N(0,1) white noise (post-warmup steps only)."""
    pipeline = _build_default_pipeline(config)
    if disabled_indices:
        nop = _NopStep()
        for idx in disabled_indices:
            if 0 <= idx < len(pipeline):
                pipeline[idx] = nop
    det = SentinelDetector(config=config, steps=pipeline)
    data = _randn_list(n, seed)
    alerts = eligible = 0
    for v in data:
        r = det.update_and_check(v)
        if not r.get('warmup', True):
            eligible += 1
            if r.get('alert', False):
                alerts += 1
    return alerts / eligible if eligible else 0.0, eligible


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — Channel Attribution
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 67)
print("1. Channel Attribution Table")
print("   Alert reasons on pure N(0,1) white noise")
print("   (n=1000, seed=99, post-warmup 970 eligible steps)")
print("=" * 67)

base_cfg = SentinelConfig()

baseline_rate, eligible = alert_rate_on_noise(base_cfg)
# EWMA channel disabled: set multiplier=100.0 so z_score never crosses threshold
ewma_off_rate, _ = alert_rate_on_noise(SentinelConfig(multiplier=100.0))
# Coherence/channel step disabled: config flag
coh_off_rate, _ = alert_rate_on_noise(SentinelConfig(enable_channel_coherence=False))
# VarCUSUM disabled: replace Step 6 (index 5) with nop
var_off_rate, _ = alert_rate_on_noise(base_cfg, disabled_indices=[5])
# Coupling disabled: config flag
coup_off_rate, _ = alert_rate_on_noise(SentinelConfig(enable_coupling_detection=False))
# Physics/maintenance disabled: Steps 26-36 = indices 25-35
phys_off_rate, _ = alert_rate_on_noise(base_cfg, disabled_indices=list(range(25, 36)))

print(f"\n{'Configuration':<42} {'Normal Alert Rate':>17}")
print("-" * 60)
print(f"{'All channels enabled (baseline)':<42} {baseline_rate*100:>15.1f}%")
print(f"{'EWMA channel disabled (multiplier=100)':<42} {ewma_off_rate*100:>15.1f}%")
print(f"{'Coherence step disabled':<42} {coh_off_rate*100:>15.1f}%")
print(f"{'Coupling detection disabled':<42} {coup_off_rate*100:>15.1f}%")
print(f"{'VarCUSUM disabled (Step 6 nop)':<42} {var_off_rate*100:>15.1f}%")
print(f"{'Physics/maintenance disabled (Steps 26-36)':<42} {phys_off_rate*100:>15.1f}%")
print(f"\nEligible post-warmup steps: {eligible}")

# Attribution deltas
print("\nContribution to false-alert rate (baseline - disabled):")
print(f"  EWMA threshold:    {(baseline_rate - ewma_off_rate)*100:+.1f} pp")
print(f"  Coherence:         {(baseline_rate - coh_off_rate)*100:+.1f} pp")
print(f"  Coupling:          {(baseline_rate - coup_off_rate)*100:+.1f} pp")
print(f"  VarCUSUM:          {(baseline_rate - var_off_rate)*100:+.1f} pp")
print(f"  Physics:           {(baseline_rate - phys_off_rate)*100:+.1f} pp")

# Alert reasons breakdown on baseline
print("\nAlert-reasons frequency on baseline (top reasons on normal noise):")
base_cfg2 = SentinelConfig()
pipeline2 = _build_default_pipeline(base_cfg2)
det2 = SentinelDetector(config=base_cfg2, steps=pipeline2)
data2 = _randn_list(1000, 99)
reason_counts = {}
total_alerts = 0
for v in data2:
    r = det2.update_and_check(v)
    if not r.get('warmup', True) and r.get('alert', False):
        total_alerts += 1
        for reason in r.get('alert_reasons', []):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
    print(f"  {reason:<40} {count:>4} ({count/eligible*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — Precision-Recall Operating Curve
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 67)
print("2. Precision-Recall Operating Curve")
print("   Varying EWMA multiplier 1.5 → 4.0  (seed=42, n=1000)")
print("=" * 67)

multipliers = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
header = f"{'Mult':>5} | {'Norm%':>6} | {'Pt F1':>6} | {'Ctx F1':>7} | {'Coll F1':>8} | {'Drift F1':>9} | {'Var F1':>7}"
print(f"\n{header}")
print("-" * len(header))

optimal_found = False
for mult in multipliers:
    cfg = SentinelConfig(multiplier=mult)
    nrate, _ = alert_rate_on_noise(cfg)
    pt   = evaluate('point',       config=cfg)['f1']
    ctx  = evaluate('contextual',  config=cfg)['f1']
    coll = evaluate('collective',  config=cfg)['f1']
    drft = evaluate('drift',       config=cfg)['f1']
    var  = evaluate('variance',    config=cfg)['f1']
    flag = " ← <5%" if nrate < 0.05 and not optimal_found else ""
    if nrate < 0.05 and not optimal_found:
        optimal_found = True
    print(f"{mult:>5.1f} | {nrate*100:>5.1f}% | {pt:>6.3f} | {ctx:>7.3f} | {coll:>8.3f} | {drft:>9.3f} | {var:>7.3f}{flag}")

if not optimal_found:
    print("\n  NOTE: No multiplier in [1.5, 4.0] achieves <5% normal alert rate.")
    print("  Searching extended range...")
    for mult in [5.0, 6.0, 7.0, 8.0, 10.0]:
        cfg = SentinelConfig(multiplier=mult)
        nrate, _ = alert_rate_on_noise(cfg)
        pt   = evaluate('point',       config=cfg)['f1']
        ctx  = evaluate('contextual',  config=cfg)['f1']
        coll = evaluate('collective',  config=cfg)['f1']
        drft = evaluate('drift',       config=cfg)['f1']
        var  = evaluate('variance',    config=cfg)['f1']
        flag = " ← <5%" if nrate < 0.05 and not optimal_found else ""
        if nrate < 0.05 and not optimal_found:
            optimal_found = True
        print(f"{mult:>5.1f} | {nrate*100:>5.1f}% | {pt:>6.3f} | {ctx:>7.3f} | {coll:>8.3f} | {drft:>9.3f} | {var:>7.3f}{flag}")
        if optimal_found:
            break


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — Normal Data Characterisation
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 67)
print("3. Normal Data Characterisation")
print("=" * 67)

print("\n(a) Pure N(0,1) white noise (seed=99, n=1000):")
print(f"    Alert rate = {baseline_rate*100:.1f}%  (eligible steps = {eligible})")

print("\n(b) Benchmark 'normal' label segments (seed=42, n=1000):")
print("    (label=0 steps only, post-warmup, per archetype)")
print()
for arch in ARCHETYPES:
    data, labels = generate(arch, n=1000, seed=42)
    det = SentinelDetector(config=SentinelConfig())
    alerts_n = eligible_n = 0
    for v, lbl in zip(data, labels):
        r = det.update_and_check(v)
        if not r.get('warmup', True) and lbl == 0:
            eligible_n += 1
            if r.get('alert', False):
                alerts_n += 1
    rate = alerts_n / eligible_n if eligible_n else 0.0
    print(f"    {arch:<12}: {rate*100:.1f}%  (eligible normal steps = {eligible_n})")

print("\n  'contextual' archetype base = 3*sin(2πi/20) + N(0,1)")
print("  (structured seasonal component — not stationary Gaussian)")
print("  All other archetypes: base signal is pure N(0,1) white noise")

# The reported 35.6% is from the full benchmark suite across all archetypes.
# Compute the aggregate (all-archetype) normal rate:
total_alerts_bench = total_eligible_bench = 0
for arch in ARCHETYPES:
    data, labels = generate(arch, n=1000, seed=42)
    det = SentinelDetector(config=SentinelConfig())
    for v, lbl in zip(data, labels):
        r = det.update_and_check(v)
        if not r.get('warmup', True) and lbl == 0:
            total_eligible_bench += 1
            if r.get('alert', False):
                total_alerts_bench += 1
agg_rate = total_alerts_bench / total_eligible_bench if total_eligible_bench else 0.0
print("\n  Aggregate normal-label rate across all 5 archetypes (benchmark suite):")
print(f"  {agg_rate*100:.1f}%  ({total_alerts_bench} alerts / {total_eligible_bench} eligible steps)")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4 — v12 Baseline False Positive Rate
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 67)
print("4. v12 Baseline Rate (VarCUSUM non-reset bug simulation)")
print("=" * 67)

class VarCUSUMStepV12(DetectorStep):
    """Reproduces v12 VarCUSUM behaviour: accumulators never reset after crossing h."""
    def __init__(self, config):
        self.cfg = config
        self.reset()

    def reset(self):
        self._s_hi = 0.0
        self._s_lo = 0.0
        self._var_ewma = 1.0
        self._warmed = False
        self._var_baseline = 0.0
        self._warmup_var_sum = 0.0
        self._warmup_var_count = 0

    def update(self, ctx) -> None:
        if ctx.is_warmup:
            ss = ctx.scratch.get("structural_snapshot")
            if ss is not None:
                self._warmup_var_sum += ss.variance
                self._warmup_var_count += 1
            return
        if not self._warmed:
            self._s_hi = 0.0
            self._s_lo = 0.0
            self._var_ewma = 0.0
            self._var_baseline = (
                self._warmup_var_sum / self._warmup_var_count
                if self._warmup_var_count > 0 else 1.0
            )
            self._warmed = True
        z = ctx.scratch.get("z_score", 0.0)
        v2 = z * z
        self._var_ewma = 0.9 * self._var_ewma + 0.1 * v2
        k = self.cfg.var_cusum_k
        self._s_hi = max(0.0, self._s_hi + v2 - k)
        if self._var_ewma > 1e-4:
            self._s_lo = max(0.0, self._s_lo + k - v2)
        else:
            self._s_lo = 0.0
        # BUG (v12): no reset after crossing — permanent alert once s_hi > h
        cusum_alert = self._s_hi > self.cfg.var_cusum_h or self._s_lo > self.cfg.var_cusum_h
        ctx.scratch["var_cusum_hi"] = self._s_hi
        ctx.scratch["var_cusum_lo"] = self._s_lo
        ctx.scratch["var_cusum_alert"] = cusum_alert
        if cusum_alert:
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True

    def state_dict(self): return {}
    def load_state(self, sd): pass


v12_cfg = SentinelConfig()
pipeline_v12 = _build_default_pipeline(v12_cfg)
pipeline_v12[5] = VarCUSUMStepV12(v12_cfg)  # index 5 = Step 6 = VarCUSUMStep
det_v12 = SentinelDetector(config=v12_cfg, steps=pipeline_v12)
data_noise = _randn_list(1000, 99)
alerts_v12 = eligible_v12 = 0
for v in data_noise:
    r = det_v12.update_and_check(v)
    if not r.get('warmup', True):
        eligible_v12 += 1
        if r.get('alert', False):
            alerts_v12 += 1
v12_rate = alerts_v12 / eligible_v12 if eligible_v12 else 0.0

print(f"\n  v12  (non-reset bug, simulated): {v12_rate*100:.1f}%")
print(f"  v12.1 (fixed, baseline above):   {baseline_rate*100:.1f}%")
print(f"  Improvement:                      {(v12_rate - baseline_rate)*100:.1f} pp reduction")

# Also run on benchmark seed=42 aggregate normal-label data for direct comparison
total_v12_alerts = total_v12_eligible = 0
for arch in ARCHETYPES:
    data, labels = generate(arch, n=1000, seed=42)
    pipeline_v12b = _build_default_pipeline(v12_cfg)
    pipeline_v12b[5] = VarCUSUMStepV12(v12_cfg)
    det = SentinelDetector(config=v12_cfg, steps=pipeline_v12b)
    for v, lbl in zip(data, labels):
        r = det.update_and_check(v)
        if not r.get('warmup', True) and lbl == 0:
            total_v12_eligible += 1
            if r.get('alert', False):
                total_v12_alerts += 1
v12_bench_rate = total_v12_alerts / total_v12_eligible if total_v12_eligible else 0.0
print(f"\n  v12  benchmark normal-label rate (seed=42): {v12_bench_rate*100:.1f}%")
print(f"  v12.1 benchmark normal-label rate (seed=42): {agg_rate*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Recommended Default Configuration
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 67)
print("5. Recommended Default Configuration")
print("=" * 67)
print()
print("  [Data for this section assembled from Tasks 1-4 above.]")
print()
print("  Done.")
