"""
Example 4: Auto-tuning from labeled data.

Shows how to tune detector parameters when you have historical
labeled anomaly data.
"""

import math
import random

from fracttalix import SentinelDetector, SentinelConfig

# ──────────────────────────────────────────────────────────────────────────────
# 1. Generate labeled dataset using stdlib only
#    Normal = sine wave + small noise  (label=False)
#    Anomaly = same but with large spike injected every ~25 steps (label=True)
# ──────────────────────────────────────────────────────────────────────────────

random.seed(99)
N_TOTAL = 300
FREQ = 2 * math.pi / 20
AMP = 2.0
NOISE_SD = 0.25
SPIKE_MAGNITUDE = 12.0
SPIKE_EVERY = 25    # inject a spike at every 25th step

labeled_data = []   # list of (value, is_anomaly)

for i in range(N_TOTAL):
    is_anomaly = (i > 0 and i % SPIKE_EVERY == 0)
    base = AMP * math.sin(FREQ * i) + random.gauss(0.0, NOISE_SD)
    if is_anomaly:
        value = base + SPIKE_MAGNITUDE * random.choice([-1, 1])
    else:
        value = base
    labeled_data.append((value, is_anomaly))

n_anomalies = sum(1 for _, lbl in labeled_data if lbl)
n_normal = N_TOTAL - n_anomalies
print("=" * 60)
print("Example 4: Auto-Tuning from Labeled Data")
print("=" * 60)
print(f"Dataset: {N_TOTAL} samples  ({n_anomalies} anomalies, {n_normal} normal)\n")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Baseline F1 with default production config
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_f1(detector, data):
    """Run detector on labeled data and return precision, recall, F1."""
    tp = fp = fn = 0
    for value, label in data:
        result = detector.update_and_check(value)
        pred = result["alert"] and not result["warmup"]
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif not pred and label:
            fn += 1
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return precision, recall, f1, tp, fp, fn

default_config = SentinelConfig.production()
default_det = SentinelDetector(config=default_config)
prec0, rec0, f1_0, tp0, fp0, fn0 = evaluate_f1(default_det, labeled_data)

print("Baseline (production defaults):")
print(f"  alpha={default_config.alpha}  multiplier={default_config.multiplier}")
print(f"  TP={tp0}  FP={fp0}  FN={fn0}")
print(f"  Precision={prec0:.4f}  Recall={rec0:.4f}  F1={f1_0:.4f}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Auto-tune with grid search
#    SentinelDetector.auto_tune(labeled_data=...) objective defaults to "f1"
# ──────────────────────────────────────────────────────────────────────────────

print("Running auto_tune() grid search over alpha × multiplier ...")
print("  alphas      = [0.05, 0.1, 0.2, 0.3]")
print("  multipliers = [2.0, 2.5, 3.0, 3.5, 4.0]")
print("  objective   = F1 score\n")

tuned_det = SentinelDetector.auto_tune(
    data=[v for v, _ in labeled_data],
    labeled_data=labeled_data,
    alphas=[0.05, 0.1, 0.2, 0.3],
    multipliers=[2.0, 2.5, 3.0, 3.5, 4.0],
)
tuned_config = tuned_det.config

# ──────────────────────────────────────────────────────────────────────────────
# 4. Evaluate tuned detector on the same dataset (re-run fresh)
# ──────────────────────────────────────────────────────────────────────────────

# auto_tune returns a fresh detector (has already been run during grid search);
# reset it and re-run for a clean evaluation pass.
tuned_det.reset()
prec1, rec1, f1_1, tp1, fp1, fn1 = evaluate_f1(tuned_det, labeled_data)

print("Tuned detector results:")
print(f"  alpha={tuned_config.alpha}  multiplier={tuned_config.multiplier}")
print(f"  TP={tp1}  FP={fp1}  FN={fn1}")
print(f"  Precision={prec1:.4f}  Recall={rec1:.4f}  F1={f1_1:.4f}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Before/after comparison
# ──────────────────────────────────────────────────────────────────────────────

delta_f1 = f1_1 - f1_0
print("Comparison:")
print(f"  {'Metric':>12}  {'Baseline':>10}  {'Tuned':>10}  {'Delta':>10}")
print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")
print(f"  {'Precision':>12}  {prec0:>10.4f}  {prec1:>10.4f}  {prec1-prec0:>+10.4f}")
print(f"  {'Recall':>12}  {rec0:>10.4f}  {rec1:>10.4f}  {rec1-rec0:>+10.4f}")
print(f"  {'F1':>12}  {f1_0:>10.4f}  {f1_1:>10.4f}  {delta_f1:>+10.4f}")
print()

print("Tuned config parameters:")
print(f"  alpha            = {tuned_config.alpha}")
print(f"  dev_alpha        = {tuned_config.dev_alpha}")
print(f"  multiplier       = {tuned_config.multiplier}")
print(f"  warmup_periods   = {tuned_config.warmup_periods}")
print()

if delta_f1 > 0:
    print(f"Auto-tuning improved F1 by {delta_f1:+.4f}.")
elif delta_f1 == 0:
    print("Auto-tuning found the same F1 — defaults are already optimal for this data.")
else:
    print(f"Auto-tuning result: {delta_f1:+.4f} — default config was already near-optimal.")

print()
print("Done — Example 4 complete.")
