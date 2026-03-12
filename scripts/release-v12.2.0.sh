#!/usr/bin/env bash
# release-v12.2.0.sh — one-shot script to merge, retag, and release v12.2.0
# Run from the Fracttalix repo root: bash scripts/release-v12.2.0.sh
set -euo pipefail

echo "=== Step 1/5: Create and merge PR ==="
BRANCH="claude/archive-repo-organization-e8xoV"
EXISTING=$(gh pr list --head "$BRANCH" --state open --json number --jq '.[0].number' 2>/dev/null || true)
if [ -z "$EXISTING" ]; then
  gh pr create --base main --head "$BRANCH" \
    --title "v12.2.0: Zenodo metadata, corpus integrity fixes, epistemic corrections" \
    --body "$(cat <<'BODY'
## Summary

- **Sentinel v12.2.0**: epistemic language corrections + production() multiplier 3.0→4.5 (FPR 35.6%→~6%)
- **Zenodo metadata**: .zenodo.json version 12.2.0, 162 claims, 21 AI layers
- **zenodo-dataset-description.md**: v12.2 changes, MK-P6 added, paper count 5→6
- **Corpus integrity (S55)**: P3 claim count, A-3.1 cross-ref, 18 orphan derivation_sources, bootstrap rewrite, Build Table, journal index

## Validation

- 21/21 AI layers valid, 0 errors
- 162 claims, 225+ derivation_source entries
BODY
)"
  EXISTING=$(gh pr list --head "$BRANCH" --state open --json number --jq '.[0].number')
fi
echo "PR #$EXISTING exists. Merging..."
gh pr merge "$EXISTING" --merge --delete-branch
echo "PR merged."

echo ""
echo "=== Step 2/5: Update local main ==="
git fetch origin main
git checkout main
git pull origin main

echo ""
echo "=== Step 3/5: Delete wrong v12.2.0 tag ==="
git tag -d v12.2.0 2>/dev/null || true
git push origin :refs/tags/v12.2.0 2>/dev/null || true

echo ""
echo "=== Step 4/5: Create correct v12.2.0 tag on main HEAD ==="
git tag v12.2.0
git push origin v12.2.0

echo ""
echo "=== Step 5/5: Create GitHub Release (triggers Zenodo webhook) ==="
gh release create v12.2.0 \
  --title "Fracttalix Sentinel v12.2.0" \
  --notes "$(cat <<'NOTES'
## Sentinel v12.2.0

### Epistemic Language Corrections
- Replaced "physics-derived" framing with "signal-processing heuristic" throughout
- Corrected README maintenance burden formula to match implementation (μ = 1−κ̄)
- Reframed "thermodynamic arrow" as "heuristic ordering hypothesis"

### Default Multiplier Change
- `SentinelConfig.production()` multiplier: 3.0 → 4.5
- Normal FPR: 35.6% → ~6% on white noise N(0,1)
- Users who need v12.1 behaviour: `SentinelConfig(multiplier=3.0)`

### Zenodo Metadata
- .zenodo.json updated to v12.2.0 (162 claims, 21 AI layers)
- MK-P6 added to dataset description

Full changelog: CHANGELOG.md
NOTES
)"

echo ""
echo "=== Done ==="
echo "Release created. Zenodo should pick this up automatically via webhook."
echo "Check: https://zenodo.org/doi/10.5281/zenodo.18859299"
