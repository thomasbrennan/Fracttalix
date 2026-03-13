#!/usr/bin/env python3
"""Fracttalix Team Orchestrator — Automated Coordinator Continuity.

External supervisor that runs OUTSIDE Claude Code instances. It:
  1. Launches a coordinator instance
  2. Monitors its checkpoint heartbeat (state.json modification time)
  3. When the coordinator dies (stale checkpoint), launches a replacement
  4. Passes full recovery context to the replacement
  5. Tracks generation count and enforces a restart limit

This solves the "who watches the watchmen" problem: since Claude Code
instances can die at any time without warning, something external must
detect death and trigger recovery.

Usage:
  python scripts/orchestrator.py start \\
    --session S57 \\
    --objective "Implement feature X" \\
    --team-size 4 \\
    --stale-seconds 300 \\
    --max-generations 10

  python scripts/orchestrator.py status
  python scripts/orchestrator.py stop

Session S56 — 2026-03-13
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / ".checkpoint"
STATE_FILE = CHECKPOINT_DIR / "state.json"
ORCHESTRATOR_STATE = CHECKPOINT_DIR / "orchestrator.json"
LOG_FILE = CHECKPOINT_DIR / "watchdog.log"

# --- Recovery prompt template ---

RECOVERY_PROMPT = """You are a RECOVERY COORDINATOR for the Fracttalix project (generation {generation}).

A previous coordinator instance has died (hit its ~200K token limit or crashed).
Your job is to resume operations seamlessly.

IMMEDIATE ACTIONS (do these in order):
1. Run: python scripts/checkpoint.py status
2. Read the output — it shows all tasks, their status, and progress
3. Run: python scripts/checkpoint.py validate
4. Read docs/claude-bootstrap.md for full project context if needed
5. Resume incomplete tasks — pick up where the dead instance left off

CRITICAL RULES:
- Do NOT re-initialize the checkpoint (state from the dead instance is preserved)
- Do NOT redo completed tasks — trust the checkpoint
- Resume in_progress tasks from their last checkpoint
- If workers were running, they died too — reassign their tasks
- Update state.json frequently (the watchdog monitors its modification time)
- If you go {stale_seconds}+ seconds without updating state.json, the watchdog
  will assume YOU are dead too and launch your replacement

SESSION OBJECTIVE: {objective}
TEAM SIZE: {team_size}
GENERATION: {generation} of {max_generations}

Read .checkpoint/PROTOCOL.md for the full checkpoint protocol.
"""

INITIAL_PROMPT = """You are the COORDINATOR for the Fracttalix project.

SESSION OBJECTIVE: {objective}
TEAM SIZE: {team_size}

YOUR FIRST ACTIONS:
1. Initialize the checkpoint:
   python scripts/checkpoint.py init --session {session} --role coordinator \\
     --objective "{objective}" --team-size {team_size} --force

2. Read project context:
   - docs/claude-bootstrap.md (full project state)
   - .checkpoint/PROTOCOL.md (checkpoint protocol rules)

3. Decompose the objective into tasks:
   python scripts/checkpoint.py task-add --id T-001 --desc "..." --priority N

4. If team_size > 1, launch workers using the Agent tool:
   - Assign tasks in state.json before launching workers
   - Workers should read their assignments from .checkpoint/state.json
   - Launch a verifier as the last worker

5. Update state.json frequently — the watchdog monitors it.
   If you go {stale_seconds}+ seconds without updating, the watchdog
   will assume you are dead and launch a replacement.

IMPORTANT: Commit your work early and often. You will die eventually.
Every completed task should be a git commit.
"""


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def get_state_age() -> float:
    """Seconds since state.json was last modified."""
    if not STATE_FILE.exists():
        return float("inf")
    return time.time() - STATE_FILE.stat().st_mtime


def load_orchestrator_state() -> dict:
    if ORCHESTRATOR_STATE.exists():
        with open(ORCHESTRATOR_STATE) as f:
            return json.load(f)
    return {}


def save_orchestrator_state(state: dict) -> None:
    ORCHESTRATOR_STATE.parent.mkdir(parents=True, exist_ok=True)
    with open(ORCHESTRATOR_STATE, "w") as f:
        json.dump(state, f, indent=2)


def launch_claude(prompt: str) -> subprocess.Popen:
    """Launch a Claude Code instance with the given prompt."""
    cmd = [
        "claude",
        "--print",
        prompt
    ]
    log(f"Launching: claude --print '<prompt>' (PID pending)")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    log(f"Claude instance launched (PID: {proc.pid})")
    return proc


def is_process_alive(proc: subprocess.Popen) -> bool:
    return proc.poll() is None


def cmd_start(args):
    log("=" * 60)
    log("Fracttalix Team Orchestrator starting")
    log(f"Session: {args.session}")
    log(f"Objective: {args.objective}")
    log(f"Team size: {args.team_size}")
    log(f"Stale threshold: {args.stale_seconds}s")
    log(f"Max generations: {args.max_generations}")
    log("=" * 60)

    orch_state = {
        "session": args.session,
        "objective": args.objective,
        "team_size": args.team_size,
        "stale_seconds": args.stale_seconds,
        "max_generations": args.max_generations,
        "current_generation": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "pid": os.getpid()
    }
    save_orchestrator_state(orch_state)

    generation = 0
    current_proc = None

    def shutdown(signum, frame):
        log(f"Received signal {signum}. Shutting down.")
        orch_state["status"] = "stopped"
        save_orchestrator_state(orch_state)
        if current_proc and is_process_alive(current_proc):
            log(f"Terminating Claude instance (PID: {current_proc.pid})")
            current_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while generation < args.max_generations:
        generation += 1
        orch_state["current_generation"] = generation
        save_orchestrator_state(orch_state)

        if generation == 1:
            prompt = INITIAL_PROMPT.format(
                session=args.session,
                objective=args.objective,
                team_size=args.team_size,
                stale_seconds=args.stale_seconds
            )
            log(f"Generation 1: Launching initial coordinator")
        else:
            prompt = RECOVERY_PROMPT.format(
                generation=generation,
                objective=args.objective,
                team_size=args.team_size,
                stale_seconds=args.stale_seconds,
                max_generations=args.max_generations
            )
            log(f"Generation {generation}: Launching recovery coordinator")

        current_proc = launch_claude(prompt)

        # Monitor loop: watch for process death OR stale checkpoint
        while True:
            time.sleep(10)  # Check every 10 seconds

            # Case 1: Process exited cleanly
            if not is_process_alive(current_proc):
                exit_code = current_proc.returncode
                log(f"Claude instance exited (code: {exit_code})")

                # Check if work is complete
                if STATE_FILE.exists():
                    with open(STATE_FILE) as f:
                        state = json.load(f)
                    tasks = state.get("task_graph", [])
                    all_done = all(t["status"] == "completed" for t in tasks) if tasks else False
                    has_handoff = "handoff" in state

                    if all_done or has_handoff:
                        log("All tasks completed or handoff prepared. Orchestrator done.")
                        orch_state["status"] = "completed"
                        save_orchestrator_state(orch_state)
                        return

                log("Instance exited but work incomplete. Launching replacement...")
                break  # Launch next generation

            # Case 2: Process alive but checkpoint stale
            age = get_state_age()
            if age > args.stale_seconds and generation > 0:
                # Only kill on staleness if we actually have a state file
                # (generation 1 might not have created it yet)
                if STATE_FILE.exists() or age > args.stale_seconds * 3:
                    log(f"Checkpoint stale ({age:.0f}s > {args.stale_seconds}s threshold)")
                    log("Coordinator presumed unresponsive. Terminating...")
                    current_proc.terminate()
                    try:
                        current_proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        current_proc.kill()
                    break  # Launch next generation

    log(f"Max generations ({args.max_generations}) reached. Manual intervention required.")
    orch_state["status"] = "exhausted"
    save_orchestrator_state(orch_state)


def cmd_status(args):
    orch_state = load_orchestrator_state()
    if not orch_state:
        print("No orchestrator state found. Run 'start' first.")
        return

    print(f"=== Orchestrator Status ===")
    print(f"Session:      {orch_state.get('session', '?')}")
    print(f"Status:       {orch_state.get('status', '?')}")
    print(f"Generation:   {orch_state.get('current_generation', 0)}/{orch_state.get('max_generations', '?')}")
    print(f"Objective:    {orch_state.get('objective', '?')}")
    print(f"Team size:    {orch_state.get('team_size', '?')}")
    print(f"Started:      {orch_state.get('started_at', '?')}")
    print(f"PID:          {orch_state.get('pid', '?')}")

    if STATE_FILE.exists():
        age = get_state_age()
        print(f"\nCheckpoint age: {age:.0f}s")
        with open(STATE_FILE) as f:
            state = json.load(f)
        tasks = state.get("task_graph", [])
        done = sum(1 for t in tasks if t["status"] == "completed")
        print(f"Tasks:        {done}/{len(tasks)} completed")


def cmd_stop(args):
    orch_state = load_orchestrator_state()
    if not orch_state:
        print("No orchestrator state found.")
        return

    pid = orch_state.get("pid")
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"Sent SIGTERM to orchestrator (PID: {pid})")
        except ProcessLookupError:
            print(f"Orchestrator process (PID: {pid}) not found — already stopped.")

    orch_state["status"] = "stopped"
    save_orchestrator_state(orch_state)


def main():
    parser = argparse.ArgumentParser(description="Fracttalix Team Orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("start", help="Start the orchestrator")
    p.add_argument("--session", required=True, help="Session ID (e.g., S57)")
    p.add_argument("--objective", required=True, help="What this session should accomplish")
    p.add_argument("--team-size", type=int, default=1)
    p.add_argument("--stale-seconds", type=int, default=300,
                   help="Seconds before checkpoint is considered stale (default: 300)")
    p.add_argument("--max-generations", type=int, default=10,
                   help="Max coordinator restarts before giving up (default: 10)")

    sub.add_parser("status", help="Show orchestrator status")
    sub.add_parser("stop", help="Stop the orchestrator")

    args = parser.parse_args()
    {"start": cmd_start, "status": cmd_status, "stop": cmd_stop}[args.command](args)


if __name__ == "__main__":
    main()
