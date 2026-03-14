#!/usr/bin/env python3
"""
Hub-and-spoke message bus using filesystem-based JSON.

All messages route through the coordinator (hub). Executors and verifier
are spokes. The module enforces that spoke-to-spoke messages are rejected;
spokes may only communicate with the coordinator.

File layout:
    .checkpoint/messages/{seq:06d}_{from}_{to}.json
    .checkpoint/messages/_seq.json              (sequence counter)

Usage:
    python scripts/message_bus.py send --from coordinator --to executor-1 --type task_assign --payload '{"task_id":"T-001"}'
    python scripts/message_bus.py read --for executor-1
    python scripts/message_bus.py read --for executor-1 --since 5
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Checkpoint integration
# ---------------------------------------------------------------------------


def _checkpoint_on_send(msg: Dict[str, Any]) -> None:
    """Auto-checkpoint after every message send for mortality-aware persistence."""
    try:
        from checkpoint import load_state, save_state  # noqa: delayed import
    except ImportError:
        # Also try relative import from scripts/
        try:
            _scripts = Path(__file__).resolve().parent
            if str(_scripts) not in sys.path:
                sys.path.insert(0, str(_scripts))
            from checkpoint import load_state, save_state
        except ImportError:
            return  # checkpoint module not available — skip silently

    try:
        state = load_state()
    except SystemExit:
        return  # no checkpoint state yet

    # Append to a rolling log of recent messages (keep last 50)
    log = state["session_state"].setdefault("message_log", [])
    log.append({
        "seq": msg["seq"],
        "from": msg["from"],
        "to": msg["to"],
        "type": msg["type"],
        "timestamp": msg["timestamp"],
    })
    if len(log) > 50:
        state["session_state"]["message_log"] = log[-50:]
    save_state(state)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
MSG_DIR = _REPO_ROOT / ".checkpoint" / "messages"
SEQ_FILE = MSG_DIR / "_seq.json"

COORDINATOR = "coordinator"
ALL_ROLES = frozenset({
    "coordinator",
    "executor-1",
    "executor-2",
    "executor-3",
    "verifier",
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dir() -> None:
    """Create the messages directory if it does not exist."""
    MSG_DIR.mkdir(parents=True, exist_ok=True)


def _next_seq() -> int:
    """Atomically increment and return the next sequence number.

    Uses ``fcntl.flock`` on ``_seq.json`` to guarantee mutual exclusion
    across concurrent processes.
    """
    _ensure_dir()

    # Open (or create) the counter file in r+/w mode
    if not SEQ_FILE.exists():
        SEQ_FILE.write_text(json.dumps({"seq": 0}))

    fd = os.open(str(SEQ_FILE), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        data = os.read(fd, 4096)
        current = json.loads(data)["seq"] if data else 0
        next_val = current + 1
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({"seq": next_val}).encode())
        return next_val
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _validate_routing(from_role: str, to_role: str) -> None:
    """Enforce hub-and-spoke: spokes may only talk to the coordinator."""
    if from_role not in ALL_ROLES:
        raise ValueError(f"Unknown sender role: {from_role}")
    if to_role not in ALL_ROLES:
        raise ValueError(f"Unknown recipient role: {to_role}")
    if from_role != COORDINATOR and to_role != COORDINATOR:
        raise ValueError(
            f"Hub-and-spoke violation: {from_role} -> {to_role}. "
            "Spokes must communicate through the coordinator."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def send_message(
    from_role: str,
    to_role: str,
    msg_type: str,
    payload: Any,
) -> Dict[str, Any]:
    """Send a message from *from_role* to *to_role*.

    Returns the written message dict (including its sequence number).
    """
    _validate_routing(from_role, to_role)
    _ensure_dir()

    seq = _next_seq()
    msg = {
        "seq": seq,
        "from": from_role,
        "to": to_role,
        "type": msg_type,
        "payload": payload,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    filename = f"{seq:06d}_{from_role}_{to_role}.json"
    filepath = MSG_DIR / filename
    filepath.write_text(json.dumps(msg, indent=2) + "\n")
    _checkpoint_on_send(msg)
    return msg


def read_messages(
    for_role: str,
    since_seq: int = 0,
) -> List[Dict[str, Any]]:
    """Read all messages addressed to *for_role* with seq > *since_seq*.

    Returns a list of message dicts sorted by sequence number.
    """
    _ensure_dir()
    results: List[Dict[str, Any]] = []

    for entry in MSG_DIR.iterdir():
        if entry.name.startswith("_") or not entry.name.endswith(".json"):
            continue
        try:
            msg = json.loads(entry.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if msg.get("to") == for_role and msg.get("seq", 0) > since_seq:
            results.append(msg)

    results.sort(key=lambda m: m["seq"])
    return results


def broadcast(
    from_role: str,
    msg_type: str,
    payload: Any,
) -> List[Dict[str, Any]]:
    """Broadcast a message from *from_role* to every *other* role.

    Only the coordinator may broadcast (hub-and-spoke constraint).
    Returns the list of sent messages.
    """
    if from_role != COORDINATOR:
        raise ValueError(
            f"Only the coordinator may broadcast (got: {from_role})."
        )
    sent: List[Dict[str, Any]] = []
    for role in sorted(ALL_ROLES):
        if role == from_role:
            continue
        sent.append(send_message(from_role, role, msg_type, payload))
    return sent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filesystem-based hub-and-spoke message bus",
    )
    sub = parser.add_subparsers(dest="command")

    # -- send --
    p_send = sub.add_parser("send", help="Send a message")
    p_send.add_argument("--from", dest="from_role", required=True)
    p_send.add_argument("--to", dest="to_role", required=True)
    p_send.add_argument("--type", dest="msg_type", required=True)
    p_send.add_argument(
        "--payload",
        default="{}",
        help="JSON string (default: '{}')",
    )

    # -- read --
    p_read = sub.add_parser("read", help="Read messages for a role")
    p_read.add_argument("--for", dest="for_role", required=True)
    p_read.add_argument("--since", dest="since_seq", type=int, default=0)

    # -- broadcast --
    p_bcast = sub.add_parser("broadcast", help="Broadcast from coordinator")
    p_bcast.add_argument("--from", dest="from_role", default=COORDINATOR)
    p_bcast.add_argument("--type", dest="msg_type", required=True)
    p_bcast.add_argument("--payload", default="{}")

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "send":
        payload = json.loads(args.payload)
        msg = send_message(args.from_role, args.to_role, args.msg_type, payload)
        print(json.dumps(msg, indent=2))

    elif args.command == "read":
        msgs = read_messages(args.for_role, args.since_seq)
        print(json.dumps(msgs, indent=2))

    elif args.command == "broadcast":
        payload = json.loads(args.payload)
        msgs = broadcast(args.from_role, args.msg_type, payload)
        print(json.dumps(msgs, indent=2))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
