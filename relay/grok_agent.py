#!/usr/bin/env python3
"""Grok Relay Agent — autonomous message processing via xAI API.

Reads pending messages addressed to Grok from relay/queue/,
sends them to the xAI API for processing, and commits responses
back to the queue. Designed to run in GitHub Actions or locally.

Usage:
    python -m relay.grok_agent              # Process all pending Grok messages
    python -m relay.grok_agent --dry-run    # Show what would be sent, don't call API
    python -m relay.grok_agent --once MSG-xxx  # Process a single message

Requires:
    XAI_API_KEY environment variable (or GitHub secret)
"""
from __future__ import annotations

import json
import os
import random
import string
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

REPO_ROOT = Path(__file__).resolve().parent.parent
RELAY_DIR = REPO_ROOT / "relay"
QUEUE_DIR = RELAY_DIR / "queue"
BOOTSTRAP_PATH = RELAY_DIR / "grok-bootstrap.md"

XAI_API_URL = "https://api.x.ai/v1/chat/completions"
XAI_MODEL = "grok-4.20-beta-0309-reasoning"

# Response type mapping
RESPONSE_TYPE_MAP = {
    "qc-request": "qc-response",
    "claim-review": "claim-review-response",
    "cross-reference": "cross-reference-response",
    "status-query": "status-response",
}


def get_api_key() -> str:
    """Get xAI API key from environment."""
    key = os.environ.get("XAI_API_KEY", "")
    if not key:
        print("ERROR: XAI_API_KEY environment variable not set.", file=sys.stderr)
        print("Get your key at: https://console.x.ai/", file=sys.stderr)
        sys.exit(1)
    return key


def generate_message_id() -> str:
    """Generate a unique message ID."""
    now = datetime.now(timezone.utc)
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"MSG-{now.strftime('%Y%m%d-%H%M%S')}-{suffix}"


def load_bootstrap() -> str:
    """Load the Grok bootstrap context."""
    return BOOTSTRAP_PATH.read_text()


def get_pending_grok_messages() -> list[tuple[Path, dict]]:
    """Get all pending messages addressed to Grok."""
    pending = []
    for path in sorted(QUEUE_DIR.glob("MSG-*.json")):
        with open(path) as f:
            msg = json.load(f)
        if msg.get("to") in ("grok", "all") and msg.get("status") == "pending":
            pending.append((path, msg))
    return pending


def call_xai_api(system_prompt: str, user_message: str, api_key: str) -> str:
    """Call the xAI API and return Grok's response."""
    payload = json.dumps({
        "model": XAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,
    }).encode()

    req = Request(
        XAI_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    import time
    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            with urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"]
        except HTTPError as e:
            body = e.read().decode() if e.fp else ""
            print(f"API error {e.code}: {body}", file=sys.stderr)
            print(f"  URL: {XAI_API_URL}", file=sys.stderr)
            print(f"  Model: {XAI_MODEL}", file=sys.stderr)
            print(f"  Key prefix: {api_key[:8]}...{api_key[-4:]}", file=sys.stderr)
            if attempt < max_retries and e.code in (403, 429, 500, 502, 503):
                wait = 2 ** (attempt + 1)
                print(f"  Retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                # Rebuild request since urlopen consumes it
                req = Request(
                    XAI_API_URL,
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                    method="POST",
                )
                continue
            raise


def build_system_prompt() -> str:
    """Build the system prompt from the bootstrap doc."""
    bootstrap = load_bootstrap()
    return f"""{bootstrap}

IMPORTANT INSTRUCTIONS:
- You are Grok, acting as an independent reviewer for the Fracttalix project.
- You will receive relay messages containing review requests.
- Respond with a JSON object containing your review.
- Your JSON response MUST include these fields:
  "reviewed_claim": the claim ID (e.g. "F-1.4"),
  "verdict": one of "confirmed", "disputed", "inconclusive", "needs-revision",
  "confidence": a number from 0.0 to 1.0,
  "reasoning": your detailed analysis,
  "sources_checked": list of sources you referenced,
  "suggestions": any improvements or concerns
- Be adversarial — catch errors, don't rubber-stamp.
- If the message is not a claim review (e.g. a status query or general message),
  respond naturally but still in JSON format with at minimum "response" and "status" fields.
- Output ONLY the JSON. No markdown fences, no preamble."""


def build_user_message(msg: dict) -> str:
    """Build the user message from a relay message."""
    parts = [
        f"Subject: {msg['subject']}",
        f"From: {msg['from']}",
        f"Type: {msg['type']}",
        f"Priority: {msg.get('priority', 'normal')}",
        "",
        msg["body"],
    ]

    refs = msg.get("references", {})
    if refs.get("claim_ids"):
        parts.append(f"\nClaim IDs: {', '.join(refs['claim_ids'])}")
    if refs.get("files"):
        parts.append(f"Files: {', '.join(refs['files'])}")

    return "\n".join(parts)


def parse_grok_response(raw: str, original_msg: dict) -> dict:
    """Parse Grok's response into a relay message."""
    # Try to extract JSON from the response
    text = raw.strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        review = json.loads(text)
    except json.JSONDecodeError:
        # If Grok didn't return valid JSON, wrap the raw text
        review = {
            "response": text,
            "status": "parse-error",
            "note": "Grok's response was not valid JSON; raw text preserved.",
        }

    # Build the relay response message
    response_type = RESPONSE_TYPE_MAP.get(original_msg["type"], "general")

    # Build body from the review
    if "verdict" in review:
        body_parts = [
            f"**Verdict**: {review['verdict']}",
            f"**Confidence**: {review.get('confidence', 'N/A')}",
            "",
            review.get("reasoning", ""),
        ]
        if review.get("sources_checked"):
            body_parts.append(f"\n**Sources**: {'; '.join(review['sources_checked'])}")
        if review.get("suggestions"):
            body_parts.append(f"\n**Suggestion**: {review['suggestions']}")
        body = "\n".join(body_parts)
    else:
        body = review.get("response", json.dumps(review, indent=2))

    response_msg = {
        "id": generate_message_id(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "from": "grok",
        "to": original_msg["from"],
        "type": response_type,
        "priority": original_msg.get("priority", "normal"),
        "subject": f"Re: {original_msg['subject']}",
        "body": body,
        "references": {
            "parent_message": original_msg["id"],
        },
        "grok_raw_review": review,
        "status": "pending",
    }

    # Carry over claim IDs
    if original_msg.get("references", {}).get("claim_ids"):
        response_msg["references"]["claim_ids"] = original_msg["references"]["claim_ids"]

    return response_msg


def save_response(msg: dict) -> Path:
    """Save a response message to the queue."""
    path = QUEUE_DIR / f"{msg['id']}.json"
    with open(path, "w") as f:
        json.dump(msg, f, indent=2)
    return path


def mark_original_resolved(path: Path) -> None:
    """Mark the original message as resolved."""
    with open(path) as f:
        msg = json.load(f)
    msg["status"] = "resolved"
    with open(path, "w") as f:
        json.dump(msg, f, indent=2)


def git_commit_and_push(files: list[str], message: str) -> bool:
    """Commit changed files and push."""
    try:
        for f in files:
            subprocess.run(["git", "add", f], check=True, cwd=REPO_ROOT)
        subprocess.run(
            ["git", "commit", "-m", message],
            check=True, cwd=REPO_ROOT,
        )
        subprocess.run(
            ["git", "push"],
            check=True, cwd=REPO_ROOT,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Git error: {e}", file=sys.stderr)
        return False


def process_message(
    path: Path,
    msg: dict,
    api_key: str,
    system_prompt: str,
    *,
    dry_run: bool = False,
) -> dict | None:
    """Process a single message through the xAI API."""
    print(f"\nProcessing: {msg['id']}")
    print(f"  Type: {msg['type']}")
    print(f"  Subject: {msg['subject']}")

    user_message = build_user_message(msg)

    if dry_run:
        print(f"  [DRY RUN] Would send to xAI API:")
        print(f"  Model: {XAI_MODEL}")
        print(f"  Message length: {len(user_message)} chars")
        return None

    print(f"  Sending to xAI API ({XAI_MODEL})...")
    raw_response = call_xai_api(system_prompt, user_message, api_key)
    print(f"  Response received ({len(raw_response)} chars)")

    response_msg = parse_grok_response(raw_response, msg)
    response_path = save_response(response_msg)
    mark_original_resolved(path)

    print(f"  Response saved: {response_msg['id']}")
    if "verdict" in response_msg.get("grok_raw_review", {}):
        review = response_msg["grok_raw_review"]
        print(f"  Verdict: {review['verdict']} ({review.get('confidence', '?')})")

    return response_msg


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Grok Relay Agent")
    parser.add_argument("--dry-run", action="store_true", help="Don't call API")
    parser.add_argument("--once", default=None, help="Process single message ID")
    parser.add_argument("--no-push", action="store_true", help="Don't git push")
    args = parser.parse_args()

    api_key = "" if args.dry_run else get_api_key()
    system_prompt = build_system_prompt()

    # Get pending messages
    if args.once:
        path = QUEUE_DIR / f"{args.once}.json"
        if not path.exists():
            print(f"Message not found: {args.once}")
            return 1
        with open(path) as f:
            msg = json.load(f)
        pending = [(path, msg)]
    else:
        pending = get_pending_grok_messages()

    if not pending:
        print("No pending messages for Grok.")
        return 0

    print(f"Found {len(pending)} pending message(s) for Grok.")

    changed_files = []
    results = []

    for path, msg in pending:
        result = process_message(path, msg, api_key, system_prompt, dry_run=args.dry_run)
        if result:
            changed_files.append(str(path))
            changed_files.append(str(QUEUE_DIR / f"{result['id']}.json"))
            results.append(result)

    if not args.dry_run and changed_files and not args.no_push:
        verdicts = []
        for r in results:
            review = r.get("grok_raw_review", {})
            v = review.get("verdict", "?")
            c = review.get("reviewed_claim", r.get("references", {}).get("claim_ids", ["?"])[0] if r.get("references", {}).get("claim_ids") else "?")
            verdicts.append(f"{c}: {v}")

        summary = "; ".join(verdicts) if verdicts else "processed"
        commit_msg = f"relay(grok-agent): auto-review — {summary}"
        git_commit_and_push(changed_files, commit_msg)

    print(f"\nDone. Processed {len(results)} message(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
