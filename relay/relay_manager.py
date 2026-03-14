#!/usr/bin/env python3
"""Multi-AI Relay Manager — git-based message relay for cross-AI communication.

Enables Claude and Grok (and future agents) to exchange structured messages
through the Fracttalix repository, supporting quality control cross-referencing,
claim verification, and general communication.

Usage:
    python -m relay.relay_manager send --to grok --type qc-request --subject "Review F-1.1" --body "..."
    python -m relay.relay_manager inbox [--agent claude]
    python -m relay.relay_manager read MSG-20260314-120000-a1b2
    python -m relay.relay_manager respond MSG-20260314-120000-a1b2 --verdict confirmed --body "..."
    python -m relay.relay_manager status
    python -m relay.relay_manager archive
    python -m relay.relay_manager qc-review --claims F-1.1,F-1.2 --to grok
"""
from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RELAY_DIR = REPO_ROOT / "relay"
QUEUE_DIR = RELAY_DIR / "queue"
ARCHIVE_DIR = RELAY_DIR / "messages"
PROTOCOL_PATH = RELAY_DIR / "protocol.json"

VALID_AGENTS = {"claude", "grok", "human", "all"}
VALID_TYPES = {
    "qc-request", "qc-response",
    "claim-review", "claim-review-response",
    "cross-reference", "cross-reference-response",
    "status-query", "status-response",
    "general",
}
VALID_PRIORITIES = {"low", "normal", "high", "critical"}
VALID_VERDICTS = {"confirmed", "disputed", "inconclusive", "needs-revision"}
VALID_STATUSES = {"pending", "acknowledged", "in-progress", "resolved", "expired"}

# Response type mapping
RESPONSE_TYPE_MAP = {
    "qc-request": "qc-response",
    "claim-review": "claim-review-response",
    "cross-reference": "cross-reference-response",
    "status-query": "status-response",
}


def generate_message_id() -> str:
    """Generate a unique message ID: MSG-YYYYMMDD-HHMMSS-xxxx."""
    now = datetime.now(timezone.utc)
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"MSG-{now.strftime('%Y%m%d-%H%M%S')}-{suffix}"


def load_protocol() -> dict:
    """Load the relay protocol definition."""
    with open(PROTOCOL_PATH) as f:
        return json.load(f)


def create_message(
    *,
    from_agent: str,
    to_agent: str,
    msg_type: str,
    subject: str,
    body: str,
    priority: str = "normal",
    claim_ids: list[str] | None = None,
    files: list[str] | None = None,
    commits: list[str] | None = None,
    parent_message: str | None = None,
) -> dict:
    """Create a structured relay message."""
    msg = {
        "id": generate_message_id(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "from": from_agent,
        "to": to_agent,
        "type": msg_type,
        "priority": priority,
        "subject": subject,
        "body": body,
        "references": {},
        "status": "pending",
    }

    if claim_ids:
        msg["references"]["claim_ids"] = claim_ids
    if files:
        msg["references"]["files"] = files
    if commits:
        msg["references"]["commits"] = commits
    if parent_message:
        msg["references"]["parent_message"] = parent_message

    return msg


def save_message(msg: dict) -> Path:
    """Save a message to the queue directory."""
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    path = QUEUE_DIR / f"{msg['id']}.json"
    with open(path, "w") as f:
        json.dump(msg, f, indent=2)
    return path


def list_queue(agent_filter: str | None = None) -> list[dict]:
    """List all messages in the queue, optionally filtered by recipient."""
    messages = []
    for path in sorted(QUEUE_DIR.glob("MSG-*.json")):
        with open(path) as f:
            msg = json.load(f)
        if agent_filter and msg.get("to") not in (agent_filter, "all"):
            continue
        messages.append(msg)
    return messages


def read_message(msg_id: str) -> dict | None:
    """Read a specific message by ID."""
    path = QUEUE_DIR / f"{msg_id}.json"
    if not path.exists():
        # Check archive
        path = ARCHIVE_DIR / f"{msg_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def update_message_status(msg_id: str, status: str) -> bool:
    """Update the status of a queued message."""
    path = QUEUE_DIR / f"{msg_id}.json"
    if not path.exists():
        return False
    with open(path) as f:
        msg = json.load(f)
    msg["status"] = status
    with open(path, "w") as f:
        json.dump(msg, f, indent=2)
    return True


def archive_resolved() -> list[str]:
    """Move resolved messages from queue to archive."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archived = []
    for path in QUEUE_DIR.glob("MSG-*.json"):
        with open(path) as f:
            msg = json.load(f)
        if msg.get("status") in ("resolved", "expired"):
            dest = ARCHIVE_DIR / path.name
            path.rename(dest)
            archived.append(msg["id"])
    return archived


def get_relay_status() -> dict:
    """Get overall relay status."""
    queue_msgs = list(QUEUE_DIR.glob("MSG-*.json"))
    archive_msgs = list(ARCHIVE_DIR.glob("MSG-*.json"))

    pending = 0
    by_agent = {}
    by_type = {}

    for path in queue_msgs:
        with open(path) as f:
            msg = json.load(f)
        if msg.get("status") == "pending":
            pending += 1
        to = msg.get("to", "unknown")
        by_agent[to] = by_agent.get(to, 0) + 1
        mtype = msg.get("type", "unknown")
        by_type[mtype] = by_type.get(mtype, 0) + 1

    protocol = load_protocol()
    agents = list(protocol.get("agents", {}).keys())

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agents_registered": agents,
        "queue_depth": len(queue_msgs),
        "pending": pending,
        "archived": len(archive_msgs),
        "by_recipient": by_agent,
        "by_type": by_type,
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

def cmd_send(args: argparse.Namespace) -> int:
    """Send a message to another agent."""
    claim_ids = args.claims.split(",") if args.claims else None
    files = args.files.split(",") if args.files else None

    msg = create_message(
        from_agent=args.sender,
        to_agent=args.to,
        msg_type=args.type,
        subject=args.subject,
        body=args.body,
        priority=args.priority,
        claim_ids=claim_ids,
        files=files,
        parent_message=args.reply_to,
    )
    path = save_message(msg)
    print(f"Message sent: {msg['id']}")
    print(f"  From: {msg['from']} → To: {msg['to']}")
    print(f"  Type: {msg['type']} ({msg['priority']})")
    print(f"  Subject: {msg['subject']}")
    print(f"  Saved: {path.relative_to(REPO_ROOT)}")
    print()
    print("Next: commit and push to make this message visible to the recipient.")
    return 0


def cmd_inbox(args: argparse.Namespace) -> int:
    """Show inbox for an agent."""
    messages = list_queue(args.agent)
    if not messages:
        print(f"No messages in queue for {args.agent or 'all agents'}.")
        return 0

    print(f"Inbox for: {args.agent or 'all agents'}")
    print(f"{'─' * 72}")
    for msg in messages:
        status_icon = {
            "pending": "●",
            "acknowledged": "◐",
            "in-progress": "◑",
            "resolved": "○",
            "expired": "×",
        }.get(msg["status"], "?")
        priority_mark = " !" if msg.get("priority") == "high" else " !!" if msg.get("priority") == "critical" else ""
        print(f"  {status_icon} [{msg['id']}] {msg['from']}→{msg['to']}{priority_mark}")
        print(f"    {msg['type']}: {msg['subject']}")
        print(f"    Status: {msg['status']} | {msg['timestamp'][:19]}")
        print()
    print(f"Total: {len(messages)} message(s)")
    return 0


def cmd_read(args: argparse.Namespace) -> int:
    """Read a specific message."""
    msg = read_message(args.message_id)
    if not msg:
        print(f"Message not found: {args.message_id}")
        return 1
    print(json.dumps(msg, indent=2))
    return 0


def cmd_respond(args: argparse.Namespace) -> int:
    """Respond to a message."""
    original = read_message(args.message_id)
    if not original:
        print(f"Message not found: {args.message_id}")
        return 1

    response_type = RESPONSE_TYPE_MAP.get(original["type"], "general")

    body = args.body
    if args.verdict:
        body = f"**Verdict**: {args.verdict}\n**Confidence**: {args.confidence}\n\n{body}"

    response = create_message(
        from_agent=args.sender,
        to_agent=original["from"],
        msg_type=response_type,
        subject=f"Re: {original['subject']}",
        body=body,
        parent_message=original["id"],
        claim_ids=original.get("references", {}).get("claim_ids"),
    )
    save_message(response)

    # Mark original as resolved
    update_message_status(args.message_id, "resolved")

    print(f"Response sent: {response['id']}")
    print(f"  In reply to: {original['id']}")
    print(f"  Original marked: resolved")
    return 0


def cmd_qc_review(args: argparse.Namespace) -> int:
    """Create a quality-control review request."""
    claim_ids = args.claims.split(",") if args.claims else []
    files = args.files.split(",") if args.files else []

    # Build a structured QC body
    lines = ["## Quality Control Review Request\n"]
    if claim_ids:
        lines.append("### Claims to verify")
        for cid in claim_ids:
            lines.append(f"- `{cid}`")
        lines.append("")
    if files:
        lines.append("### Files to review")
        for f in files:
            lines.append(f"- `{f}`")
        lines.append("")
    if args.context:
        lines.append("### Context")
        lines.append(args.context)
        lines.append("")

    lines.append("### Requested actions")
    lines.append("1. Independent verification of listed claims against source material")
    lines.append("2. Cross-reference with external literature/databases")
    lines.append("3. Flag any inconsistencies, errors, or gaps")
    lines.append("4. Provide confidence level and reasoning")

    msg = create_message(
        from_agent=args.sender,
        to_agent=args.to,
        msg_type="qc-request",
        subject=f"QC Review: {', '.join(claim_ids) if claim_ids else 'general review'}",
        body="\n".join(lines),
        priority=args.priority,
        claim_ids=claim_ids,
        files=files,
    )
    save_message(msg)
    print(f"QC Review request created: {msg['id']}")
    print(f"  To: {msg['to']}")
    print(f"  Claims: {', '.join(claim_ids) if claim_ids else 'none'}")
    print(f"  Files: {', '.join(files) if files else 'none'}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show relay status."""
    status = get_relay_status()

    if args.json:
        print(json.dumps(status, indent=2))
        return 0

    print(f"[{status['timestamp'][:19]}] Multi-AI Relay Status")
    print(f"  Agents: {', '.join(status['agents_registered'])}")
    print(f"  Queue depth: {status['queue_depth']} ({status['pending']} pending)")
    print(f"  Archived: {status['archived']}")
    if status["by_recipient"]:
        print(f"  By recipient: {status['by_recipient']}")
    if status["by_type"]:
        print(f"  By type: {status['by_type']}")
    return 0


def cmd_check_inbox(args: argparse.Namespace) -> int:
    """Session-start inbox check — designed for autonomous pickup.

    Called at the beginning of each AI session to see if there are
    pending messages. Returns exit code 0 if messages found, 1 if empty.
    This lets session hooks branch on whether there's relay work to do.
    """
    agent = args.agent
    messages = list_queue(agent)
    pending = [m for m in messages if m.get("status") == "pending"]

    if not pending:
        print(f"No pending messages for {agent}.")
        return 1

    print(f"You have {len(pending)} pending message(s):\n")
    for msg in pending:
        priority_mark = ""
        if msg.get("priority") == "high":
            priority_mark = " [HIGH]"
        elif msg.get("priority") == "critical":
            priority_mark = " [CRITICAL]"

        print(f"  From: {msg['from']}{priority_mark}")
        print(f"  Type: {msg['type']}")
        print(f"  Subject: {msg['subject']}")
        print(f"  ID: {msg['id']}")
        print(f"  Time: {msg['timestamp'][:19]}")

        # Show a preview of the body (first 3 lines)
        body_lines = msg.get("body", "").strip().split("\n")
        preview = "\n    ".join(body_lines[:3])
        if len(body_lines) > 3:
            preview += "\n    ..."
        print(f"  Preview:\n    {preview}")
        print()

    print(f"Use 'python -m relay.relay_manager read <MSG-ID>' to read full message.")
    print(f"Use 'python -m relay.relay_manager respond <MSG-ID> --body \"...\"' to respond.")
    return 0


def cmd_archive(args: argparse.Namespace) -> int:
    """Archive resolved messages."""
    archived = archive_resolved()
    if archived:
        print(f"Archived {len(archived)} message(s):")
        for mid in archived:
            print(f"  {mid}")
    else:
        print("No resolved messages to archive.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="relay_manager",
        description="Multi-AI Relay Manager — git-based cross-AI communication",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # send
    p_send = sub.add_parser("send", help="Send a message")
    p_send.add_argument("--from", dest="sender", default="claude", choices=sorted(VALID_AGENTS - {"all"}))
    p_send.add_argument("--to", required=True, choices=sorted(VALID_AGENTS))
    p_send.add_argument("--type", required=True, choices=sorted(VALID_TYPES))
    p_send.add_argument("--subject", required=True)
    p_send.add_argument("--body", required=True)
    p_send.add_argument("--priority", default="normal", choices=sorted(VALID_PRIORITIES))
    p_send.add_argument("--claims", default=None, help="Comma-separated claim IDs")
    p_send.add_argument("--files", default=None, help="Comma-separated file paths")
    p_send.add_argument("--reply-to", default=None, help="Parent message ID")
    p_send.set_defaults(func=cmd_send)

    # inbox
    p_inbox = sub.add_parser("inbox", help="Show inbox")
    p_inbox.add_argument("--agent", default=None, help="Filter by recipient agent")
    p_inbox.set_defaults(func=cmd_inbox)

    # read
    p_read = sub.add_parser("read", help="Read a message")
    p_read.add_argument("message_id")
    p_read.set_defaults(func=cmd_read)

    # respond
    p_respond = sub.add_parser("respond", help="Respond to a message")
    p_respond.add_argument("message_id")
    p_respond.add_argument("--from", dest="sender", default="claude", choices=sorted(VALID_AGENTS - {"all"}))
    p_respond.add_argument("--body", required=True)
    p_respond.add_argument("--verdict", default=None, choices=sorted(VALID_VERDICTS))
    p_respond.add_argument("--confidence", type=float, default=0.5)
    p_respond.set_defaults(func=cmd_respond)

    # qc-review
    p_qc = sub.add_parser("qc-review", help="Create QC review request")
    p_qc.add_argument("--from", dest="sender", default="claude", choices=sorted(VALID_AGENTS - {"all"}))
    p_qc.add_argument("--to", default="grok", choices=sorted(VALID_AGENTS))
    p_qc.add_argument("--claims", default=None, help="Comma-separated claim IDs")
    p_qc.add_argument("--files", default=None, help="Comma-separated file paths")
    p_qc.add_argument("--context", default=None, help="Additional context")
    p_qc.add_argument("--priority", default="normal", choices=sorted(VALID_PRIORITIES))
    p_qc.set_defaults(func=cmd_qc_review)

    # status
    p_status = sub.add_parser("status", help="Relay status")
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=cmd_status)

    # check-inbox (session start)
    p_check = sub.add_parser("check-inbox", help="Session-start inbox check")
    p_check.add_argument("--agent", default="claude", help="Which agent to check for")
    p_check.set_defaults(func=cmd_check_inbox)

    # archive
    p_archive = sub.add_parser("archive", help="Archive resolved messages")
    p_archive.set_defaults(func=cmd_archive)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
