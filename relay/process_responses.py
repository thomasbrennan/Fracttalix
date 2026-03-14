#!/usr/bin/env python3
"""Process Grok responses — parse verdicts and update review tracker.

Scans the relay queue for Grok responses, extracts verdicts,
and updates the review tracker with coverage data.

Usage:
    python -m relay.process_responses          # Process all unprocessed responses
    python -m relay.process_responses --report  # Generate a review report
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUEUE_DIR = REPO_ROOT / "relay" / "queue"
ARCHIVE_DIR = REPO_ROOT / "relay" / "messages"
TRACKER_FILE = REPO_ROOT / "relay" / "review-tracker.json"
REPORT_FILE = REPO_ROOT / "relay" / "review-report.md"


def load_tracker() -> dict:
    """Load the review tracker."""
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
    with open(TRACKER_FILE, "w") as f:
        json.dump(tracker, f, indent=2)


def find_grok_responses() -> list[dict]:
    """Find all Grok response messages in queue and archive."""
    responses = []
    for directory in (QUEUE_DIR, ARCHIVE_DIR):
        for path in directory.glob("MSG-*.json"):
            with open(path) as f:
                msg = json.load(f)
            if msg.get("from") == "grok" and "grok_raw_review" in msg:
                responses.append(msg)
    return responses


def process_responses() -> dict:
    """Process all Grok responses and update tracker."""
    tracker = load_tracker()
    responses = find_grok_responses()
    new_reviews = 0

    for msg in responses:
        review = msg.get("grok_raw_review", {})
        claim_ids = msg.get("references", {}).get("claim_ids", [])
        parent_id = msg.get("references", {}).get("parent_message")

        # Extract verdict info
        verdict = review.get("verdict")
        confidence = review.get("confidence")
        reviewed_claim = review.get("reviewed_claim")

        # Determine which claim this reviews
        target_claims = []
        if reviewed_claim:
            target_claims = [reviewed_claim]
        elif claim_ids:
            target_claims = claim_ids

        for cid in target_claims:
            if cid not in tracker["claims_reviewed"]:
                new_reviews += 1
            tracker["claims_reviewed"][cid] = {
                "msg_id": msg["id"],
                "parent_msg_id": parent_id,
                "timestamp": msg.get("timestamp"),
                "verdict": verdict,
                "confidence": confidence,
                "model_used": msg.get("_model_used", "unknown"),
                "reasoning_preview": (review.get("reasoning", "") or "")[:200],
            }

    # Update summary counts
    verdicts = {}
    for cid, info in tracker["claims_reviewed"].items():
        v = info.get("verdict", "unknown")
        verdicts[v] = verdicts.get(v, 0) + 1

    tracker["summary"]["reviewed_by_grok"] = len(tracker["claims_reviewed"])
    tracker["summary"]["confirmed"] = verdicts.get("confirmed", 0)
    tracker["summary"]["disputed"] = verdicts.get("disputed", 0)
    tracker["summary"]["inconclusive"] = verdicts.get("inconclusive", 0)
    tracker["summary"]["needs_revision"] = verdicts.get("needs-revision", 0) + verdicts.get("needs_revision", 0)

    save_tracker(tracker)
    return {"new_reviews": new_reviews, "total_reviewed": len(tracker["claims_reviewed"])}


def generate_report() -> str:
    """Generate a markdown review report."""
    tracker = load_tracker()
    s = tracker["summary"]
    reviewed = tracker.get("claims_reviewed", {})

    lines = [
        "# Grok Review Report",
        "",
        f"**Generated:** {__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()[:19]}Z",
        "",
        "## Summary",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Total falsifiable | {s.get('total_falsifiable', '?')} |",
        f"| Sent to Grok | {s.get('sent_to_grok', 0)} |",
        f"| Reviewed by Grok | {s.get('reviewed_by_grok', 0)} |",
        f"| Confirmed | {s.get('confirmed', 0)} |",
        f"| Disputed | {s.get('disputed', 0)} |",
        f"| Inconclusive | {s.get('inconclusive', 0)} |",
        f"| Needs revision | {s.get('needs_revision', 0)} |",
        "",
    ]

    # Disputed claims section (most important)
    disputed = {cid: info for cid, info in reviewed.items() if info.get("verdict") == "disputed"}
    if disputed:
        lines.extend([
            "## Disputed Claims (Action Required)",
            "",
        ])
        for cid, info in sorted(disputed.items()):
            lines.extend([
                f"### {cid}",
                f"- **Confidence:** {info.get('confidence', '?')}",
                f"- **Model:** {info.get('model_used', '?')}",
                f"- **Reasoning:** {info.get('reasoning_preview', 'N/A')}",
                f"- **Message:** {info.get('msg_id', '?')}",
                "",
            ])

    # Needs revision
    needs_rev = {cid: info for cid, info in reviewed.items() if info.get("verdict") == "needs-revision"}
    if needs_rev:
        lines.extend([
            "## Needs Revision",
            "",
        ])
        for cid, info in sorted(needs_rev.items()):
            lines.extend([
                f"- **{cid}** (confidence: {info.get('confidence', '?')}): {info.get('reasoning_preview', '')}",
            ])
        lines.append("")

    # Confirmed claims
    confirmed = {cid: info for cid, info in reviewed.items() if info.get("verdict") == "confirmed"}
    if confirmed:
        lines.extend([
            "## Confirmed Claims",
            "",
            "| Claim | Confidence | Model |",
            "|-------|-----------|-------|",
        ])
        for cid, info in sorted(confirmed.items()):
            lines.append(f"| {cid} | {info.get('confidence', '?')} | {info.get('model_used', '?')} |")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Process Grok responses")
    parser.add_argument("--report", action="store_true", help="Generate review report")
    args = parser.parse_args()

    result = process_responses()
    print(f"Processed {result['new_reviews']} new review(s). Total reviewed: {result['total_reviewed']}")

    if args.report:
        report = generate_report()
        with open(REPORT_FILE, "w") as f:
            f.write(report)
        print(f"Report written to: {REPORT_FILE.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
