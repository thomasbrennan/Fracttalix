#!/usr/bin/env python3
"""Team role registry with heartbeats.

Manages role registration, heartbeat tracking, and team status
under .checkpoint/team/{role_name}.json.

Usage:
    python scripts/team_registry.py register --role coordinator --capabilities plan,assign,checkpoint
    python scripts/team_registry.py heartbeat --role coordinator
    python scripts/team_registry.py status
    python scripts/team_registry.py deregister --role coordinator
    python scripts/team_registry.py is-alive --role coordinator [--timeout 60]
"""

import argparse
import fcntl
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TEAM_DIR = Path(__file__).resolve().parent.parent / ".checkpoint" / "team"


# ---------------------------------------------------------------------------
# Checkpoint integration
# ---------------------------------------------------------------------------


def _checkpoint_on_role_change(role_name: str, event: str) -> None:
    """Auto-checkpoint after role register/deregister for mortality-aware persistence."""
    try:
        _scripts = Path(__file__).resolve().parent
        if str(_scripts) not in sys.path:
            sys.path.insert(0, str(_scripts))
        from checkpoint import load_state, save_state
    except ImportError:
        return

    try:
        state = load_state()
    except SystemExit:
        return  # no checkpoint state yet

    log = state["session_state"].setdefault("role_events", [])
    log.append({
        "role": role_name,
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    if len(log) > 30:
        state["session_state"]["role_events"] = log[-30:]

    # Update team_size from live registry
    team = get_team()
    state["_meta"]["team_size"] = len(team)
    save_state(state)


# Default capabilities per role kind
ROLE_DEFAULTS = {
    "coordinator": ["plan", "assign", "checkpoint"],
    "executor": ["code", "test", "commit"],
    "verifier": ["review", "validate", "approve"],
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _role_path(role_name: str) -> Path:
    return TEAM_DIR / f"{role_name}.json"


def _write_json(path: Path, data: dict) -> None:
    """Write JSON atomically with flock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, json.dumps(data, indent=2).encode() + b"\n")
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _read_json(path: Path) -> dict:
    """Read JSON with shared flock."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        raw = b""
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            raw += chunk
        return json.loads(raw)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def register_role(role_name: str, capabilities: list, pid: int = None) -> dict:
    """Register a role, writing its JSON file under .checkpoint/team/."""
    now = _now_iso()
    data = {
        "role": role_name,
        "capabilities": capabilities,
        "pid": pid if pid is not None else os.getpid(),
        "registered_at": now,
        "last_heartbeat": now,
        "status": "active",
    }
    _write_json(_role_path(role_name), data)
    _checkpoint_on_role_change(role_name, "registered")
    return data


def heartbeat(role_name: str) -> dict:
    """Update the last_heartbeat timestamp for a role."""
    path = _role_path(role_name)
    if not path.exists():
        raise FileNotFoundError(f"Role '{role_name}' is not registered")
    data = _read_json(path)
    data["last_heartbeat"] = _now_iso()
    data["status"] = "active"
    _write_json(path, data)
    return data


def get_team() -> dict:
    """Return dict of all registered roles keyed by role name."""
    TEAM_DIR.mkdir(parents=True, exist_ok=True)
    team = {}
    for f in sorted(TEAM_DIR.glob("*.json")):
        try:
            data = _read_json(f)
            role_name = data.get("role", f.stem)
            team[role_name] = data
        except (json.JSONDecodeError, OSError):
            continue
    return team


def is_alive(role_name: str, timeout: int = 60) -> bool:
    """Check whether a role's heartbeat is within timeout seconds."""
    path = _role_path(role_name)
    if not path.exists():
        return False
    data = _read_json(path)
    last = datetime.fromisoformat(data["last_heartbeat"])
    age = (datetime.now(timezone.utc) - last).total_seconds()
    return age <= timeout


def deregister(role_name: str) -> None:
    """Remove a role's JSON file."""
    path = _role_path(role_name)
    if path.exists():
        path.unlink()
        _checkpoint_on_role_change(role_name, "deregistered")


# --------------- CLI ---------------

def _cli_register(args):
    caps = args.capabilities.split(",") if args.capabilities else ROLE_DEFAULTS.get(args.role, [])
    data = register_role(args.role, caps, pid=args.pid)
    print(json.dumps(data, indent=2))


def _cli_heartbeat(args):
    data = heartbeat(args.role)
    print(f"Heartbeat updated for {args.role}: {data['last_heartbeat']}")


def _cli_status(_args):
    team = get_team()
    if not team:
        print("No roles registered.")
        return
    print(f"{'Role':<20} {'Status':<10} {'PID':<8} {'Last Heartbeat':<30} {'Capabilities'}")
    print("-" * 100)
    for name, info in team.items():
        alive = is_alive(name)
        display_status = "active" if alive else "stale"
        caps = ",".join(info.get("capabilities", []))
        print(f"{name:<20} {display_status:<10} {info.get('pid','?'):<8} {info.get('last_heartbeat','?'):<30} {caps}")


def _cli_deregister(args):
    deregister(args.role)
    print(f"Deregistered {args.role}")


def _cli_is_alive(args):
    alive = is_alive(args.role, timeout=args.timeout)
    status = "alive" if alive else "dead/missing"
    print(f"{args.role}: {status}")
    sys.exit(0 if alive else 1)


def main():
    parser = argparse.ArgumentParser(description="Team role registry")
    sub = parser.add_subparsers(dest="command")

    p_reg = sub.add_parser("register", help="Register a role")
    p_reg.add_argument("--role", required=True)
    p_reg.add_argument("--capabilities", default=None, help="Comma-separated capabilities")
    p_reg.add_argument("--pid", type=int, default=None)

    p_hb = sub.add_parser("heartbeat", help="Send heartbeat")
    p_hb.add_argument("--role", required=True)

    sub.add_parser("status", help="Show team overview")

    p_dereg = sub.add_parser("deregister", help="Remove a role")
    p_dereg.add_argument("--role", required=True)

    p_alive = sub.add_parser("is-alive", help="Check if role is alive")
    p_alive.add_argument("--role", required=True)
    p_alive.add_argument("--timeout", type=int, default=60)

    args = parser.parse_args()
    dispatch = {
        "register": _cli_register,
        "heartbeat": _cli_heartbeat,
        "status": _cli_status,
        "deregister": _cli_deregister,
        "is-alive": _cli_is_alive,
    }
    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
