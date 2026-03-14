#!/usr/bin/env python3
"""Automated QC Pipeline — scans AI layers and queues Grok reviews.

Modes:
    full-scan   — Queue ALL unreviewed falsifiable claims for Grok review
    diff-scan   — Queue only claims in files changed since last scan
    single      — Queue a specific claim or paper for review
    status      — Show review coverage across the corpus

Usage:
    python -m relay.auto_qc full-scan [--dry-run] [--batch-size N]
    python -m relay.auto_qc diff-scan [--base-ref HEAD~1]
    python -m relay.auto_qc single --paper P1 [--claim F-1.4]
    python -m relay.auto_qc status
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from relay.relay_manager import create_message, save_message, list_queue
from relay.cost_router import select_model, estimate_message_cost, check_budget, get_remaining_budget

REPO_ROOT = Path(__file__).resolve().parent.parent
AI_LAYERS_DIR = REPO_ROOT / "ai-layers"
QUEUE_DIR = REPO_ROOT / "relay" / "queue"
TRACKER_FILE = REPO_ROOT / "relay" / "review-tracker.json"


def load_ai_layer(path: Path) -> dict | None:
    """Load and validate an AI layer JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def get_all_ai_layers() -> list[tuple[Path, dict]]:
    """Load all AI layer files."""
    layers = []
    for path in sorted(AI_LAYERS_DIR.glob("*-ai-layer.json")):
        data = load_ai_layer(path)
        if data and "claim_registry" in data:
            layers.append((path, data))
    return layers


def extract_falsifiable_claims(layer_data: dict) -> list[dict]:
    """Extract all falsifiable claims from an AI layer."""
    claims = []
    for claim in layer_data.get("claim_registry", []):
        if claim.get("type") == "F" and claim.get("falsification_predicate"):
            claims.append(claim)
    return claims


def extract_all_claims(layer_data: dict) -> list[dict]:
    """Extract all claims (any type) from an AI layer."""
    return layer_data.get("claim_registry", [])


def load_tracker() -> dict:
    """Load the review tracker (which claims have been sent/reviewed by Grok)."""
    if TRACKER_FILE.exists():
        with open(TRACKER_FILE) as f:
            return json.load(f)
    return {
        "version": "1.0",
        "last_scan": None,
        "claims_sent": {},
        "claims_reviewed": {},
        "summary": {
            "total_falsifiable": 0,
            "sent_to_grok": 0,
            "reviewed_by_grok": 0,
            "confirmed": 0,
            "disputed": 0,
            "inconclusive": 0,
            "needs_revision": 0,
        },
    }


def save_tracker(tracker: dict) -> None:
    """Save the review tracker."""
    with open(TRACKER_FILE, "w") as f:
        json.dump(tracker, f, indent=2)


def get_pending_grok_claim_ids() -> set[str]:
    """Get claim IDs that are already queued and pending for Grok."""
    pending = set()
    for msg in list_queue("grok"):
        if msg.get("status") == "pending" and msg.get("type") in ("claim-review", "qc-request"):
            for cid in msg.get("references", {}).get("claim_ids", []):
                pending.add(cid)
    return pending


def build_claim_review_body(claim: dict, layer_data: dict) -> str:
    """Build a structured review request body for a claim."""
    paper_id = layer_data.get("paper_id", "?")
    paper_title = layer_data.get("paper_title", "?")

    parts = [
        f"## Claim Review Request: {claim['claim_id']}",
        f"**Paper:** {paper_id} — {paper_title}",
        f"**Claim:** {claim.get('name', 'unnamed')}",
        f"**Type:** {claim['type']}",
        "",
        f"### Statement",
        claim.get("statement", "(no statement)"),
        "",
    ]

    fp = claim.get("falsification_predicate")
    if fp:
        parts.append("### Falsification Predicate")
        if isinstance(fp, dict):
            for key in ("FALSIFIED_IF", "WHERE", "EVALUATION", "BOUNDARY", "CONTEXT"):
                val = fp.get(key, fp.get(key.lower()))
                if val:
                    parts.append(f"**{key}:** {val}")
            parts.append("")
        elif isinstance(fp, str):
            parts.append(fp)
            parts.append("")

    # Add derivation chain if available
    sources = claim.get("derivation_source", [])
    if sources:
        parts.append(f"**Derivation chain:** {' → '.join(sources)}")
        parts.append("")

    # Add scope boundary from the layer
    scope = layer_data.get("scope_boundary", {})
    if scope:
        parts.append("### Scope")
        parts.append(f"- In scope: {scope.get('in_scope', '?')}")
        parts.append(f"- Out of scope: {scope.get('out_of_scope', '?')}")
        parts.append("")

    parts.extend([
        "### Requested",
        "1. Verify the falsification predicate is logically consistent",
        "2. Check math against known results (Strogatz, Kuznetsov, etc.)",
        "3. Identify counterexamples or edge cases missed",
        "4. Cross-reference against published literature",
        "5. Provide verdict with confidence and reasoning",
    ])

    return "\n".join(parts)


def queue_claim_review(
    claim: dict,
    layer_data: dict,
    *,
    priority: str = "normal",
    dry_run: bool = False,
) -> dict | None:
    """Queue a single claim for Grok review. Returns the message or None."""
    paper_id = layer_data.get("paper_id", "?")
    body = build_claim_review_body(claim, layer_data)

    # Cost-aware model selection
    model = select_model("claim-review", priority)
    cost_est = estimate_message_cost(len(body) + 3000, len(body), "quality" if "latest" in model["id"] else "fast")

    if not check_budget(cost_est):
        print(f"  SKIP {claim['claim_id']}: insufficient budget (need ${cost_est:.4f}, have ${get_remaining_budget():.2f})")
        return None

    if dry_run:
        print(f"  [DRY RUN] Would queue {claim['claim_id']} ({paper_id}) → {model['id']} (~${cost_est:.4f})")
        return None

    msg = create_message(
        from_agent="claude",
        to_agent="grok",
        msg_type="claim-review",
        subject=f"Review {claim['claim_id']}: {claim.get('name', 'unnamed')}",
        body=body,
        priority=priority,
        claim_ids=[claim["claim_id"]],
        files=[f"ai-layers/{paper_id}-ai-layer.json"],
    )
    # Tag with model routing hint
    msg["_model_tier"] = "quality" if "latest" in model["id"] else "fast"
    save_message(msg)
    return msg


def cmd_full_scan(args: argparse.Namespace) -> int:
    """Full scan: queue all unreviewed falsifiable claims."""
    tracker = load_tracker()
    already_sent = set(tracker.get("claims_sent", {}).keys())
    already_pending = get_pending_grok_claim_ids()
    skip = already_sent | already_pending

    layers = get_all_ai_layers()
    total_f = 0
    queued = 0
    skipped = 0

    print(f"Scanning {len(layers)} AI layers...")
    print(f"Budget remaining: ${get_remaining_budget():.2f}")
    print(f"Already sent/pending: {len(skip)} claims")
    print()

    for path, data in layers:
        paper_id = data.get("paper_id", path.stem)
        claims = extract_falsifiable_claims(data)
        total_f += len(claims)

        if not claims:
            continue

        print(f"{paper_id}: {len(claims)} falsifiable claim(s)")

        for claim in claims:
            cid = claim["claim_id"]
            if cid in skip:
                skipped += 1
                continue

            if args.batch_size and queued >= args.batch_size:
                print(f"\nBatch limit reached ({args.batch_size}). Run again for more.")
                break

            msg = queue_claim_review(claim, data, dry_run=args.dry_run)
            if msg:
                queued += 1
                tracker["claims_sent"][cid] = {
                    "msg_id": msg["id"],
                    "timestamp": msg["timestamp"],
                    "paper_id": paper_id,
                }
                print(f"  Queued: {cid} → {msg['id']}")

        if args.batch_size and queued >= args.batch_size:
            break

    tracker["last_scan"] = datetime.now(timezone.utc).isoformat()
    tracker["summary"]["total_falsifiable"] = total_f
    tracker["summary"]["sent_to_grok"] = len(tracker["claims_sent"])

    if not args.dry_run:
        save_tracker(tracker)

    print(f"\nScan complete:")
    print(f"  Total falsifiable claims: {total_f}")
    print(f"  Skipped (already sent/pending): {skipped}")
    print(f"  Newly queued: {queued}")
    print(f"  Budget remaining: ${get_remaining_budget():.2f}")

    if queued > 0 and not args.dry_run:
        print(f"\nNext: commit and push to trigger Grok processing.")
    return 0


def cmd_diff_scan(args: argparse.Namespace) -> int:
    """Diff scan: queue claims from recently changed AI layers."""
    base_ref = args.base_ref

    # Get changed AI layer files
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base_ref, "--", "ai-layers/*-ai-layer.json"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        changed_files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    except subprocess.CalledProcessError:
        print("Failed to get git diff. Falling back to full scan.")
        args.batch_size = 10
        return cmd_full_scan(args)

    if not changed_files:
        print("No AI layer changes detected.")
        return 0

    print(f"Changed AI layers: {', '.join(changed_files)}")
    tracker = load_tracker()
    queued = 0

    for rel_path in changed_files:
        path = REPO_ROOT / rel_path
        data = load_ai_layer(path)
        if not data:
            continue

        paper_id = data.get("paper_id", path.stem)
        claims = extract_falsifiable_claims(data)

        for claim in claims:
            cid = claim["claim_id"]
            # Re-queue even if previously sent — the claim changed
            msg = queue_claim_review(claim, data, priority="high", dry_run=args.dry_run)
            if msg:
                queued += 1
                tracker["claims_sent"][cid] = {
                    "msg_id": msg["id"],
                    "timestamp": msg["timestamp"],
                    "paper_id": paper_id,
                    "trigger": "diff-scan",
                }

    if not args.dry_run:
        save_tracker(tracker)

    print(f"\nDiff scan: queued {queued} claim(s) for re-review.")
    return 0


def cmd_single(args: argparse.Namespace) -> int:
    """Queue a specific paper or claim for review."""
    paper_id = args.paper
    target_claim = args.claim

    # Find the AI layer
    layer_path = AI_LAYERS_DIR / f"{paper_id}-ai-layer.json"
    if not layer_path.exists():
        print(f"AI layer not found: {layer_path}")
        return 1

    data = load_ai_layer(layer_path)
    if not data:
        print(f"Failed to load: {layer_path}")
        return 1

    if target_claim:
        # Single claim
        claims = [c for c in data.get("claim_registry", []) if c["claim_id"] == target_claim]
        if not claims:
            print(f"Claim {target_claim} not found in {paper_id}")
            return 1
    else:
        # All falsifiable claims in the paper
        claims = extract_falsifiable_claims(data)

    tracker = load_tracker()
    queued = 0

    for claim in claims:
        msg = queue_claim_review(claim, data, priority=args.priority, dry_run=args.dry_run)
        if msg:
            queued += 1
            tracker["claims_sent"][claim["claim_id"]] = {
                "msg_id": msg["id"],
                "timestamp": msg["timestamp"],
                "paper_id": paper_id,
            }

    if not args.dry_run:
        save_tracker(tracker)

    print(f"Queued {queued} claim(s) from {paper_id}.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show review coverage status."""
    tracker = load_tracker()
    layers = get_all_ai_layers()

    # Count all falsifiable claims
    all_falsifiable = {}
    for path, data in layers:
        paper_id = data.get("paper_id", path.stem)
        claims = extract_falsifiable_claims(data)
        for c in claims:
            all_falsifiable[c["claim_id"]] = paper_id

    total = len(all_falsifiable)
    sent = set(tracker.get("claims_sent", {}).keys())
    reviewed = set(tracker.get("claims_reviewed", {}).keys())
    not_sent = set(all_falsifiable.keys()) - sent

    # Get verdicts
    verdicts = {}
    for cid, info in tracker.get("claims_reviewed", {}).items():
        v = info.get("verdict", "unknown")
        verdicts[v] = verdicts.get(v, 0) + 1

    # Budget
    remaining = get_remaining_budget()
    cost_per_review = 0.024  # approximate
    reviews_possible = int(remaining / cost_per_review)

    print(f"{'═' * 60}")
    print(f"  GROK REVIEW COVERAGE DASHBOARD")
    print(f"{'═' * 60}")
    print(f"  Total falsifiable claims:  {total}")
    print(f"  Sent to Grok:              {len(sent)}  ({len(sent)/total*100:.0f}%)" if total else "  Sent to Grok:              0")
    print(f"  Reviewed by Grok:          {len(reviewed)}  ({len(reviewed)/total*100:.0f}%)" if total else "  Reviewed by Grok:          0")
    print(f"  Not yet sent:              {len(not_sent)}")
    print()

    if verdicts:
        print(f"  Verdicts:")
        for v, count in sorted(verdicts.items()):
            print(f"    {v}: {count}")
        print()

    print(f"  Budget remaining:          ${remaining:.2f}")
    print(f"  Estimated reviews left:    ~{reviews_possible}")
    print(f"  Last scan:                 {tracker.get('last_scan', 'never')}")
    print(f"{'═' * 60}")

    # Per-paper breakdown
    if args.verbose:
        print(f"\n  Per-paper breakdown:")
        papers = {}
        for cid, pid in all_falsifiable.items():
            if pid not in papers:
                papers[pid] = {"total": 0, "sent": 0, "reviewed": 0}
            papers[pid]["total"] += 1
            if cid in sent:
                papers[pid]["sent"] += 1
            if cid in reviewed:
                papers[pid]["reviewed"] += 1

        for pid in sorted(papers):
            p = papers[pid]
            bar_len = 20
            filled = int(p["reviewed"] / p["total"] * bar_len) if p["total"] else 0
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"    {pid:12s} [{bar}] {p['reviewed']}/{p['total']}")

    # List unreviewed
    if not_sent and args.verbose:
        print(f"\n  Unreviewed claims:")
        for cid in sorted(not_sent):
            print(f"    - {cid} ({all_falsifiable[cid]})")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Automated QC Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # full-scan
    p_full = sub.add_parser("full-scan", help="Queue all unreviewed falsifiable claims")
    p_full.add_argument("--dry-run", action="store_true")
    p_full.add_argument("--batch-size", type=int, default=None, help="Max claims to queue per run")
    p_full.set_defaults(func=cmd_full_scan)

    # diff-scan
    p_diff = sub.add_parser("diff-scan", help="Queue claims from changed AI layers")
    p_diff.add_argument("--base-ref", default="HEAD~1", help="Git ref to diff against")
    p_diff.add_argument("--dry-run", action="store_true")
    p_diff.set_defaults(func=cmd_diff_scan)

    # single
    p_single = sub.add_parser("single", help="Queue specific paper/claim")
    p_single.add_argument("--paper", required=True, help="Paper ID (e.g. P1)")
    p_single.add_argument("--claim", default=None, help="Specific claim ID")
    p_single.add_argument("--priority", default="normal", choices=["low", "normal", "high", "critical"])
    p_single.add_argument("--dry-run", action="store_true")
    p_single.set_defaults(func=cmd_single)

    # status
    p_status = sub.add_parser("status", help="Show review coverage")
    p_status.add_argument("--verbose", "-v", action="store_true")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
