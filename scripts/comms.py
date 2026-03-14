#!/usr/bin/env python3
"""Git-based cross-instance communication network.

All Claude instances (Code sessions, Claude.ai, local CLI) share one thing:
the git repository. This module uses git as the transport layer for a
discovery + messaging network that works across all environments.

Architecture:
  .comms/
  ├── roster/           ← Instance self-registration (who's alive)
  │   └── {instance_id}.json
  ├── inbox/            ← Point-to-point messages
  │   └── {seq}_{from}_{to}.json
  ├── broadcast/        ← Broadcast messages (all instances read)
  │   └── {seq}_{from}_all.json
  └── _state.json       ← Global sequence counter

Transport: git add + commit + push. Instances poll via git pull.
Discovery: scan .comms/roster/ for registered instances.

Session S57 — 2026-03-13
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
# Comms live in .comms/ directory, committed to the working branch.
# All comms commits use [comms] prefix for easy filtering.
# Each instance pushes to its own branch; cross-branch discovery uses
# git fetch + read from remote refs.
COMMS_DIR = _REPO_ROOT / ".comms"
ROSTER_DIR = COMMS_DIR / "roster"
INBOX_DIR = COMMS_DIR / "inbox"
BROADCAST_DIR = COMMS_DIR / "broadcast"
SEQ_FILE = COMMS_DIR / "_state.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    for d in (ROSTER_DIR, INBOX_DIR, BROADCAST_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _next_seq() -> int:
    """Increment global sequence counter. Simple file-based — git merge
    conflicts on this file are expected and handled by taking the max."""
    if SEQ_FILE.exists():
        data = json.loads(SEQ_FILE.read_text())
        seq = data.get("seq", 0) + 1
    else:
        seq = 1
    SEQ_FILE.write_text(json.dumps({"seq": seq}) + "\n")
    return seq


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command in the repo root."""
    return subprocess.run(
        ["git"] + list(args),
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=check,
    )


def _git_sync(files: List[str], message: str) -> bool:
    """Stage comms files, commit, and push to the current branch.
    Returns True on success."""
    try:
        branch = _current_branch()
        # Pull first to reduce merge conflicts
        _git("pull", "--rebase", "origin", branch, check=False)

        for f in files:
            _git("add", f)

        result = _git("diff", "--cached", "--quiet", check=False)
        if result.returncode == 0:
            return True  # Nothing to commit

        _git("commit", "-m", message)

        # Push with retry (exponential backoff)
        for attempt in range(4):
            result = _git("push", "-u", "origin", branch, check=False)
            if result.returncode == 0:
                return True
            import time
            time.sleep(2 ** (attempt + 1))

        return False
    except subprocess.CalledProcessError:
        return False


def _current_branch() -> str:
    result = _git("rev-parse", "--abbrev-ref", "HEAD")
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Instance Identity
# ---------------------------------------------------------------------------

def _get_instance_id() -> str:
    """Generate a stable instance ID from environment."""
    # Use session ID if available (Claude Code)
    session_id = os.environ.get("CLAUDE_CODE_SESSION_ID", "")
    if session_id:
        return f"cc-{session_id[:12]}"

    # Use remote session ID if available
    remote_id = os.environ.get("CLAUDE_CODE_REMOTE_SESSION_ID", "")
    if remote_id:
        return f"remote-{remote_id[:12]}"

    # Fallback: hostname + PID
    import socket
    return f"local-{socket.gethostname()[:8]}-{os.getpid()}"


# ---------------------------------------------------------------------------
# Roster: Discovery
# ---------------------------------------------------------------------------

def register(
    instance_id: Optional[str] = None,
    role: str = "unknown",
    capabilities: Optional[List[str]] = None,
    session: str = "",
    objective: str = "",
) -> Dict[str, Any]:
    """Register this instance in the shared roster."""
    _ensure_dirs()
    iid = instance_id or _get_instance_id()

    entry = {
        "instance_id": iid,
        "role": role,
        "capabilities": capabilities or [],
        "session": session,
        "objective": objective,
        "branch": _current_branch(),
        "registered_at": _now(),
        "last_seen": _now(),
        "status": "active",
    }

    path = ROSTER_DIR / f"{iid}.json"
    path.write_text(json.dumps(entry, indent=2) + "\n")

    _git_sync(
        [str(path.relative_to(_REPO_ROOT))],
        f"[comms] Register instance {iid} ({role})"
    )
    return entry


def heartbeat(instance_id: Optional[str] = None) -> Dict[str, Any]:
    """Update last_seen timestamp for this instance."""
    _ensure_dirs()
    iid = instance_id or _get_instance_id()
    path = ROSTER_DIR / f"{iid}.json"

    if not path.exists():
        raise FileNotFoundError(f"Instance {iid} not registered. Run register first.")

    entry = json.loads(path.read_text())
    entry["last_seen"] = _now()
    entry["status"] = "active"
    path.write_text(json.dumps(entry, indent=2) + "\n")

    _git_sync(
        [str(path.relative_to(_REPO_ROOT))],
        f"[comms] Heartbeat {iid}"
    )
    return entry


def discover(scan_remote: bool = True) -> List[Dict[str, Any]]:
    """Pull latest and return all registered instances.

    If scan_remote=True, also fetches and scans .comms/roster/ from all
    remote branches (claude/*) so instances on different branches can
    discover each other.
    """
    _ensure_dirs()
    # Pull current branch
    _git("pull", "--rebase", "origin", _current_branch(), check=False)

    seen_ids: set = set()
    instances = []

    def _add_entry(entry: Dict[str, Any], source: str = "") -> None:
        iid = entry.get("instance_id", "")
        if iid in seen_ids:
            return
        seen_ids.add(iid)
        # Calculate staleness
        last_seen = entry.get("last_seen", "")
        if last_seen:
            try:
                seen_dt = datetime.fromisoformat(last_seen)
                age = (datetime.now(timezone.utc) - seen_dt).total_seconds()
                entry["_stale_seconds"] = round(age)
                entry["_alive"] = age < 600  # 10 min threshold
            except (ValueError, TypeError):
                entry["_alive"] = False
        if source:
            entry["_source_branch"] = source
        instances.append(entry)

    # 1. Read local roster
    for path in sorted(ROSTER_DIR.glob("*.json")):
        try:
            _add_entry(json.loads(path.read_text()), _current_branch())
        except (json.JSONDecodeError, OSError):
            continue

    # 2. Scan remote branches for their roster files
    if scan_remote:
        _git("fetch", "--all", check=False)
        # List all remote branches
        result = _git("branch", "-r", "--list", "origin/claude/*", check=False)
        branches = [b.strip() for b in result.stdout.strip().split("\n") if b.strip()]

        for branch in branches:
            # Read roster files from that branch via git show
            result = _git(
                "ls-tree", "--name-only", branch, ".comms/roster/",
                check=False
            )
            if result.returncode != 0:
                continue
            for filename in result.stdout.strip().split("\n"):
                if not filename.endswith(".json"):
                    continue
                show_result = _git("show", f"{branch}:{filename}", check=False)
                if show_result.returncode == 0:
                    try:
                        short_branch = branch.replace("origin/", "")
                        _add_entry(json.loads(show_result.stdout), short_branch)
                    except (json.JSONDecodeError, ValueError):
                        continue

    instances.sort(key=lambda e: e.get("last_seen", ""))
    return instances


def deregister(instance_id: Optional[str] = None) -> None:
    """Remove this instance from the roster."""
    iid = instance_id or _get_instance_id()
    path = ROSTER_DIR / f"{iid}.json"

    if path.exists():
        # Mark as departed rather than delete (preserves history)
        entry = json.loads(path.read_text())
        entry["status"] = "departed"
        entry["departed_at"] = _now()
        path.write_text(json.dumps(entry, indent=2) + "\n")

        _git_sync(
            [str(path.relative_to(_REPO_ROOT))],
            f"[comms] Deregister instance {iid}"
        )


# ---------------------------------------------------------------------------
# Messaging: Point-to-point and Broadcast
# ---------------------------------------------------------------------------

def send(
    to_instance: str,
    msg_type: str,
    payload: Any,
    from_instance: Optional[str] = None,
) -> Dict[str, Any]:
    """Send a message to a specific instance."""
    _ensure_dirs()
    fid = from_instance or _get_instance_id()
    seq = _next_seq()

    msg = {
        "seq": seq,
        "from": fid,
        "to": to_instance,
        "type": msg_type,
        "payload": payload,
        "timestamp": _now(),
    }

    filename = f"{seq:06d}_{fid}_{to_instance}.json"
    path = INBOX_DIR / filename
    path.write_text(json.dumps(msg, indent=2) + "\n")

    _git_sync(
        [
            str(path.relative_to(_REPO_ROOT)),
            str(SEQ_FILE.relative_to(_REPO_ROOT)),
        ],
        f"[comms] {fid} → {to_instance}: {msg_type}"
    )
    return msg


def broadcast(
    msg_type: str,
    payload: Any,
    from_instance: Optional[str] = None,
) -> Dict[str, Any]:
    """Broadcast a message to all instances."""
    _ensure_dirs()
    fid = from_instance or _get_instance_id()
    seq = _next_seq()

    msg = {
        "seq": seq,
        "from": fid,
        "to": "*",
        "type": msg_type,
        "payload": payload,
        "timestamp": _now(),
    }

    filename = f"{seq:06d}_{fid}_all.json"
    path = BROADCAST_DIR / filename
    path.write_text(json.dumps(msg, indent=2) + "\n")

    _git_sync(
        [
            str(path.relative_to(_REPO_ROOT)),
            str(SEQ_FILE.relative_to(_REPO_ROOT)),
        ],
        f"[comms] {fid} → ALL: {msg_type}"
    )
    return msg


def receive(
    instance_id: Optional[str] = None,
    since_seq: int = 0,
) -> List[Dict[str, Any]]:
    """Pull and read all messages for this instance (inbox + broadcasts)."""
    _ensure_dirs()
    iid = instance_id or _get_instance_id()

    # Pull latest
    _git("pull", "--rebase", "origin", _current_branch(), check=False)

    messages = []

    # Read inbox (point-to-point)
    for path in INBOX_DIR.glob("*.json"):
        try:
            msg = json.loads(path.read_text())
            if msg.get("to") == iid and msg.get("seq", 0) > since_seq:
                messages.append(msg)
        except (json.JSONDecodeError, OSError):
            continue

    # Read broadcasts
    for path in BROADCAST_DIR.glob("*.json"):
        try:
            msg = json.loads(path.read_text())
            if msg.get("seq", 0) > since_seq and msg.get("from") != iid:
                messages.append(msg)
        except (json.JSONDecodeError, OSError):
            continue

    messages.sort(key=lambda m: m.get("seq", 0))
    return messages


# ---------------------------------------------------------------------------
# Predefined message types
# ---------------------------------------------------------------------------

# Discovery
MSG_HELLO = "hello"               # New instance announcing itself
MSG_ROLL_CALL = "roll_call"       # Request all instances to heartbeat
MSG_STATUS_REQUEST = "status_req" # Request status from specific instance

# Coordination
MSG_TASK_OFFER = "task_offer"     # Coordinator offering a task
MSG_TASK_CLAIM = "task_claim"     # Instance claiming a task
MSG_TASK_DONE = "task_done"       # Instance reporting task completion

# CBP integration
MSG_CBP_REVIEW_NEEDED = "cbp_review_needed"  # Work ready for CBP cycle
MSG_CBP_PHASE_UPDATE = "cbp_phase_update"    # CBP phase transition notification

# Sync
MSG_PULL_REQUEST = "pull_request" # Ask instances to git pull for updates
MSG_CHECKPOINT_SYNC = "checkpoint_sync"  # Checkpoint state changed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Git-based cross-instance communication network"
    )
    sub = parser.add_subparsers(dest="command")

    # -- register --
    p_reg = sub.add_parser("register", help="Register this instance")
    p_reg.add_argument("--id", dest="instance_id", default=None)
    p_reg.add_argument("--role", default="unknown")
    p_reg.add_argument("--capabilities", default="")
    p_reg.add_argument("--session", default="")
    p_reg.add_argument("--objective", default="")

    # -- heartbeat --
    p_hb = sub.add_parser("heartbeat", help="Update heartbeat")
    p_hb.add_argument("--id", dest="instance_id", default=None)

    # -- discover --
    sub.add_parser("discover", help="List all known instances")

    # -- send --
    p_send = sub.add_parser("send", help="Send message to an instance")
    p_send.add_argument("--to", required=True)
    p_send.add_argument("--type", dest="msg_type", required=True)
    p_send.add_argument("--payload", default="{}")
    p_send.add_argument("--from", dest="from_id", default=None)

    # -- broadcast --
    p_bcast = sub.add_parser("broadcast", help="Broadcast to all instances")
    p_bcast.add_argument("--type", dest="msg_type", required=True)
    p_bcast.add_argument("--payload", default="{}")
    p_bcast.add_argument("--from", dest="from_id", default=None)

    # -- receive --
    p_recv = sub.add_parser("receive", help="Read messages for this instance")
    p_recv.add_argument("--id", dest="instance_id", default=None)
    p_recv.add_argument("--since", type=int, default=0)

    # -- deregister --
    p_dereg = sub.add_parser("deregister", help="Remove instance from roster")
    p_dereg.add_argument("--id", dest="instance_id", default=None)

    # -- whoami --
    sub.add_parser("whoami", help="Show this instance's ID")

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "register":
        caps = [c.strip() for c in args.capabilities.split(",") if c.strip()]
        entry = register(args.instance_id, args.role, caps, args.session, args.objective)
        print(f"Registered: {entry['instance_id']} ({entry['role']})")
        print(json.dumps(entry, indent=2))

    elif args.command == "heartbeat":
        entry = heartbeat(args.instance_id)
        print(f"Heartbeat: {entry['instance_id']} at {entry['last_seen']}")

    elif args.command == "discover":
        instances = discover()
        if not instances:
            print("No instances registered.")
        else:
            print(f"{'ID':<30s} {'Role':<15s} {'Status':<10s} {'Branch':<40s} {'Stale(s)'}")
            print("-" * 110)
            for inst in instances:
                alive = "ALIVE" if inst.get("_alive") else "STALE"
                stale = inst.get("_stale_seconds", "?")
                print(f"{inst['instance_id']:<30s} {inst['role']:<15s} "
                      f"{alive:<10s} {inst.get('branch', '?'):<40s} {stale}")

    elif args.command == "send":
        payload = json.loads(args.payload)
        msg = send(args.to, args.msg_type, payload, args.from_id)
        print(f"Sent #{msg['seq']}: {msg['from']} → {msg['to']}: {msg['type']}")

    elif args.command == "broadcast":
        payload = json.loads(args.payload)
        msg = broadcast(args.msg_type, payload, args.from_id)
        print(f"Broadcast #{msg['seq']}: {msg['from']} → ALL: {msg['type']}")

    elif args.command == "receive":
        msgs = receive(args.instance_id, args.since)
        if not msgs:
            print("No new messages.")
        else:
            for m in msgs:
                print(f"  #{m['seq']} [{m['type']}] from {m['from']}: "
                      f"{json.dumps(m['payload'])[:80]}")

    elif args.command == "deregister":
        deregister(args.instance_id)
        print("Deregistered.")

    elif args.command == "whoami":
        print(_get_instance_id())

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
