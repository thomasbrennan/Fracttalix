"""
Example 3: Collapse dynamics and intervention detection.

Demonstrates the physics-derived collapse metrics:
- Maintenance burden μ trending toward 1.0
- Diagnostic window Δt (time-to-collapse estimate)
- Reversed sequence detection (intervention signature)
"""

import math
import random

from fracttalix import SentinelDetector, SentinelConfig

# ──────────────────────────────────────────────────────────────────────────────
# 1. Detector tuned for physics metric visibility
#    Low alpha = slow EWMA → collapse metrics are computed over longer history
# ──────────────────────────────────────────────────────────────────────────────

config = SentinelConfig(
    alpha=0.1,
    dev_alpha=0.1,
    multiplier=3.0,
    warmup_periods=30,
    rpi_window=64,
    ews_window=40,
)
det = SentinelDetector(config=config)

print("=" * 60)
print("Example 3: Collapse Dynamics & Intervention Detection")
print("=" * 60)
print(f"Config: alpha={config.alpha}, warmup={config.warmup_periods}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Phase A: Healthy baseline (steps 0–59)
#    Clean oscillation, low variance
# ──────────────────────────────────────────────────────────────────────────────

random.seed(123)
TOTAL = 300
last_result = None
_prev_regime = None

print(f"{'Step':>5} {'Value':>8} {'μ (burden)':>12} {'Regime':>22} "
      f"{'Δt steps':>10} {'Φ-κ sep':>10}")
print("-" * 80)

def _fmt(v):
    return f"{v:.4f}" if v is not None else "   N/A"

for step in range(TOTAL):
    # Degradation model:
    # Phase A (0-79):   clean sine, small noise
    # Phase B (80-159): variance doubles every 20 steps (increasing disorder)
    # Phase C (160-239): frequency drifts + amplitude decays (structural loss)
    # Phase D (240-299): near-DC collapse (signal flattening)

    if step < 80:
        noise_sd = 0.2
        freq = 2 * math.pi / 25
        amp = 3.0
    elif step < 160:
        progress = (step - 80) / 80.0
        noise_sd = 0.2 + 2.5 * progress          # variance increase
        freq = 2 * math.pi / 25
        amp = 3.0 - 0.5 * progress
    elif step < 240:
        progress = (step - 160) / 80.0
        noise_sd = 2.7 + 1.5 * progress
        freq = 2 * math.pi / (25 + 20 * progress) # frequency shift (slowing)
        amp = 2.5 * (1 - 0.6 * progress)          # amplitude decay
    else:
        progress = (step - 240) / 60.0
        noise_sd = 4.2
        freq = 2 * math.pi / 50
        amp = 1.0 * (1 - 0.8 * progress)          # near-collapse

    value = amp * math.sin(freq * step) + random.gauss(0.0, noise_sd)
    result = det.update_and_check(value)

    regime = result.get("tainter_regime", "UNKNOWN")
    mu = result.get("maintenance_burden", 0.0)
    dw = result.get_diagnostic_window()
    phi_kappa = result.get("phi_kappa_separation", 0.0)

    # Print at key steps and on regime changes
    show = (
        step % 40 == 0
        or regime != _prev_regime
        or (dw["steps"] is not None and step % 20 == 0)
    )
    if show:
        dt_str = f"{dw['steps']:.1f}" if dw["steps"] is not None else "    N/A"
        print(f"{step:>5} {value:>8.2f} {mu:>12.4f} {regime:>22} "
              f"{dt_str:>10} {phi_kappa:>10.4f}")

    _prev_regime = regime
    last_result = result

print()

# ──────────────────────────────────────────────────────────────────────────────
# 3. Report final collapse metrics
# ──────────────────────────────────────────────────────────────────────────────

print("Final collapse metrics (step 299):")
mb = last_result.get_maintenance_burden()
print(f"  maintenance_burden μ = {mb['mu']:.4f}")
print(f"  tainter_regime       = {mb['regime']}")

dw = last_result.get_diagnostic_window()
print(f"  diagnostic_window (expected)    = {_fmt(dw['steps'])} steps")
print(f"  diagnostic_window (pessimistic) = {_fmt(dw['steps_pessimistic'])} steps")
print(f"  diagnostic_window (optimistic)  = {_fmt(dw['steps_optimistic'])} steps")
print(f"  confidence                      = {dw['confidence']}")
print(f"  supercompensation_detected      = {dw['supercompensation']}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 4. phi_kappa_separation = Φ (Kuramoto order) − κ̄ (mean coupling)
#    Positive → phase coherence > coupling strength → possible external intervention
#    Negative → coupling strength > phase coherence → organic degradation
# ──────────────────────────────────────────────────────────────────────────────

phi = last_result.get("kuramoto_order", 0.0)
kappa = last_result.get("mean_coupling_strength", 0.0)
sep = last_result.get("phi_kappa_separation", 0.0)
print("Phase coherence vs coupling strength:")
print(f"  Φ (Kuramoto order)     = {phi:.4f}")
print(f"  κ̄ (mean coupling)     = {kappa:.4f}")
print(f"  Φ − κ̄ separation      = {sep:+.4f}")
if sep > 0:
    print("  Interpretation: coherence leads coupling → possible intervention signature")
else:
    print("  Interpretation: coupling leads coherence → organic degradation pattern")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 5. Intervention signature
# ──────────────────────────────────────────────────────────────────────────────

iv = last_result.get_intervention_signature()
print("Intervention signature:")
print(f"  sequence_type              = {iv['sequence_type']}")
print(f"  intervention_score         = {iv['score']:.4f}")
print(f"  reversed_sequence          = {last_result.get('reversed_sequence', False)}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 6. Print tainter_regime progression as a compact summary
# ──────────────────────────────────────────────────────────────────────────────

print("Tainter regime progression (sampled at steps 0, 50, 100, ... 299):")
history = list(det._history)
for r in history:
    s = r["step"]
    if s % 50 == 0 or s == 299:
        print(f"  step {s:>3d}: {r.get('tainter_regime', 'UNKNOWN'):>22} "
              f"μ={r.get('maintenance_burden', 0.0):.4f}  "
              f"Δt={_fmt(r.get('diagnostic_window_steps'))}")

print()
print("Done — Example 3 complete.")
