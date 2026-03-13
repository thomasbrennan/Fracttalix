#!/usr/bin/env python3
"""Fracttalix Checkpoint Protocol — Mortality-Aware State Persistence.

Provides read/write/validate operations for the .checkpoint/ directory.
Designed to be called by Claude Code instances (coordinator or solo) to:
  1. Initialize a session checkpoint
  2. Record decisions, discoveries, and task progress continuously
  3. Read prior checkpoint state for recovery after instance death
  4. Validate checkpoint integrity

Usage:
  python scripts/checkpoint.py init --session S56 --role coordinator --objective "Build team protocol"
  python scripts/checkpoint.py task-add --id T-001 --desc "Implement X" --priority 2
  python scripts/checkpoint.py task-update --id T-001 --status in_progress
  python scripts/checkpoint.py task-complete --id T-001 --commit abc1234 --summary "Done"
  python scripts/checkpoint.py decide --decision "Use JSON over prose" --rationale "Token efficiency"
  python scripts/checkpoint.py discover --text "Schema v3 requires session field"
  python scripts/checkpoint.py status
  python scripts/checkpoint.py validate
  python scripts/checkpoint.py handoff --to S57 --context "Verifier role untested"

Session S56 — 2026-03-13
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / ".checkpoint"
STATE_FILE = CHECKPOINT_DIR / "state.json"
SCHEMA_FILE = CHECKPOINT_DIR / "checkpoint-schema.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state() -> dict:
    if not STATE_FILE.exists():
        print("ERROR: No checkpoint state found. Run 'init' first.", file=sys.stderr)
        sys.exit(1)
    with open(STATE_FILE) as f:
        return json.load(f)


def save_state(state: dict) -> None:
    state["_meta"]["last_updated"] = now_iso()
    state["_meta"]["checkpoint_sequence"] += 1
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    seq = state["_meta"]["checkpoint_sequence"]
    print(f"Checkpoint #{seq} written at {state['_meta']['last_updated']}")


def cmd_init(args):
    if STATE_FILE.exists() and not args.force:
        print("WARNING: Checkpoint state already exists. Use --force to overwrite.", file=sys.stderr)
        print("To resume from existing state, use 'status' instead.", file=sys.stderr)
        sys.exit(1)

    state = {
        "_meta": {
            "schema_version": "v1-S56",
            "created_session": args.session,
            "last_updated": now_iso(),
            "last_updated_by": args.role,
            "instance_role": args.role,
            "team_size": args.team_size,
            "checkpoint_sequence": 0
        },
        "session_state": {
            "objective": args.objective,
            "context_loaded": [],
            "decisions": [],
            "discoveries": [],
            "blockers": []
        },
        "task_graph": []
    }
    save_state(state)
    print(f"Session {args.session} initialized as {args.role}.")


def cmd_task_add(args):
    state = load_state()
    existing_ids = {t["task_id"] for t in state["task_graph"]}
    if args.id in existing_ids:
        print(f"ERROR: Task {args.id} already exists.", file=sys.stderr)
        sys.exit(1)

    task = {
        "task_id": args.id,
        "description": args.desc,
        "status": "pending",
        "priority": args.priority,
        "assigned_to": args.assign or None,
        "depends_on": args.depends.split(",") if args.depends else [],
        "files_in_scope": args.files.split(",") if args.files else [],
        "acceptance_criteria": args.criteria.split("|") if args.criteria else []
    }
    state["task_graph"].append(task)
    save_state(state)
    print(f"Task {args.id} added: {args.desc}")


def cmd_task_update(args):
    state = load_state()
    for task in state["task_graph"]:
        if task["task_id"] == args.id:
            if args.status:
                task["status"] = args.status
            if args.assign:
                task["assigned_to"] = args.assign
            if args.progress:
                task.setdefault("checkpoint", {})
                task["checkpoint"]["progress_summary"] = args.progress
                task["checkpoint"]["timestamp"] = now_iso()
            if args.files_modified:
                task.setdefault("checkpoint", {})
                task["checkpoint"]["files_modified"] = args.files_modified.split(",")
            save_state(state)
            print(f"Task {args.id} updated: status={task['status']}")
            return
    print(f"ERROR: Task {args.id} not found.", file=sys.stderr)
    sys.exit(1)


def cmd_task_complete(args):
    state = load_state()
    for task in state["task_graph"]:
        if task["task_id"] == args.id:
            task["status"] = "completed"
            task["completion"] = {
                "commit_sha": args.commit or "",
                "summary": args.summary,
                "timestamp": now_iso()
            }
            # Also write to completions directory for team visibility
            comp_file = CHECKPOINT_DIR / "completions" / f"{args.id}.json"
            comp_file.parent.mkdir(parents=True, exist_ok=True)
            with open(comp_file, "w") as f:
                json.dump(task["completion"], f, indent=2)
            save_state(state)
            print(f"Task {args.id} COMPLETED.")
            return
    print(f"ERROR: Task {args.id} not found.", file=sys.stderr)
    sys.exit(1)


def cmd_decide(args):
    state = load_state()
    state["session_state"]["decisions"].append({
        "decision": args.decision,
        "rationale": args.rationale,
        "timestamp": now_iso(),
        "reversible": not args.irreversible
    })
    save_state(state)
    print(f"Decision recorded: {args.decision}")


def cmd_discover(args):
    state = load_state()
    state["session_state"]["discoveries"].append(args.text)
    save_state(state)
    print(f"Discovery recorded.")


def cmd_context(args):
    state = load_state()
    state["session_state"]["context_loaded"].append(args.file)
    save_state(state)


def cmd_blocker(args):
    state = load_state()
    state["session_state"]["blockers"].append({
        "description": args.desc,
        "blocking_task": args.task,
        "resolved": False
    })
    # Also update the task status
    for task in state["task_graph"]:
        if task["task_id"] == args.task:
            task["status"] = "blocked"
    save_state(state)
    print(f"Blocker recorded for task {args.task}: {args.desc}")


def cmd_status(args):
    state = load_state()
    meta = state["_meta"]
    ss = state["session_state"]

    print(f"=== Checkpoint Status ===")
    print(f"Session:    {meta['created_session']}")
    print(f"Role:       {meta['instance_role']}")
    print(f"Team size:  {meta['team_size']}")
    print(f"Sequence:   #{meta['checkpoint_sequence']}")
    print(f"Updated:    {meta['last_updated']}")
    print(f"Objective:  {ss['objective']}")
    print()

    tasks = state["task_graph"]
    if tasks:
        print(f"=== Tasks ({len(tasks)}) ===")
        for t in tasks:
            marker = {"pending": " ", "assigned": "→", "in_progress": "▶",
                       "blocked": "✗", "completed": "✓", "failed": "✗",
                       "abandoned": "—"}.get(t["status"], "?")
            assignee = f" [{t['assigned_to']}]" if t.get("assigned_to") else ""
            print(f"  [{marker}] {t['task_id']} (P{t['priority']}) {t['description']}{assignee}")
            if t.get("checkpoint", {}).get("progress_summary"):
                print(f"      └─ {t['checkpoint']['progress_summary']}")
        print()

        done = sum(1 for t in tasks if t["status"] == "completed")
        total = len(tasks)
        print(f"Progress: {done}/{total} tasks completed")
    else:
        print("No tasks defined yet.")

    if ss["decisions"]:
        print(f"\n=== Decisions ({len(ss['decisions'])}) ===")
        for d in ss["decisions"]:
            rev = "" if d.get("reversible", True) else " [IRREVERSIBLE]"
            print(f"  • {d['decision']}{rev}")

    if ss["discoveries"]:
        print(f"\n=== Discoveries ({len(ss['discoveries'])}) ===")
        for d in ss["discoveries"]:
            print(f"  • {d}")

    blockers = [b for b in ss.get("blockers", []) if not b.get("resolved")]
    if blockers:
        print(f"\n=== Active Blockers ({len(blockers)}) ===")
        for b in blockers:
            print(f"  ✗ {b['blocking_task']}: {b['description']}")


def cmd_handoff(args):
    state = load_state()
    incomplete = [t["task_id"] for t in state["task_graph"] if t["status"] not in ("completed",)]
    state["handoff"] = {
        "type": "graceful",
        "from_session": state["_meta"]["created_session"],
        "to_session": args.to,
        "incomplete_tasks": incomplete,
        "critical_context": args.context or "",
        "timestamp": now_iso()
    }
    save_state(state)
    print(f"Handoff prepared: {state['_meta']['created_session']} → {args.to}")
    print(f"Incomplete tasks: {len(incomplete)}")
    if incomplete:
        for tid in incomplete:
            task = next(t for t in state["task_graph"] if t["task_id"] == tid)
            print(f"  • {tid}: {task['description']} [{task['status']}]")


def cmd_validate(args):
    state = load_state()
    errors = []

    # Check required fields
    if not state.get("_meta", {}).get("schema_version"):
        errors.append("Missing _meta.schema_version")
    if not state.get("session_state", {}).get("objective"):
        errors.append("Missing session_state.objective")
    if not isinstance(state.get("task_graph"), list):
        errors.append("task_graph is not a list")

    # Check task ID uniqueness
    ids = [t["task_id"] for t in state.get("task_graph", [])]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate task IDs found")

    # Check dependency integrity
    id_set = set(ids)
    for task in state.get("task_graph", []):
        for dep in task.get("depends_on", []):
            if dep not in id_set:
                errors.append(f"Task {task['task_id']} depends on unknown task {dep}")

    # Check for orphaned completions
    completions_dir = CHECKPOINT_DIR / "completions"
    if completions_dir.exists():
        for comp_file in completions_dir.glob("*.json"):
            tid = comp_file.stem
            if tid not in id_set:
                errors.append(f"Orphaned completion file for unknown task {tid}")

    if errors:
        print(f"VALIDATION FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        seq = state["_meta"]["checkpoint_sequence"]
        print(f"Checkpoint valid. Sequence #{seq}, {len(ids)} tasks.")


def main():
    parser = argparse.ArgumentParser(
        description="Fracttalix Checkpoint Protocol — mortality-aware state persistence"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p = sub.add_parser("init", help="Initialize a new checkpoint session")
    p.add_argument("--session", required=True, help="Session ID (e.g., S56)")
    p.add_argument("--role", required=True, choices=["coordinator", "executor", "verifier", "solo"])
    p.add_argument("--objective", required=True, help="Session objective")
    p.add_argument("--team-size", type=int, default=1)
    p.add_argument("--force", action="store_true", help="Overwrite existing state")

    # task-add
    p = sub.add_parser("task-add", help="Add a task to the graph")
    p.add_argument("--id", required=True, help="Task ID (T-001 format)")
    p.add_argument("--desc", required=True, help="Task description")
    p.add_argument("--priority", type=int, default=3, help="Priority 1-5 (1=critical)")
    p.add_argument("--assign", help="Assigned instance")
    p.add_argument("--depends", help="Comma-separated dependency task IDs")
    p.add_argument("--files", help="Comma-separated files in scope")
    p.add_argument("--criteria", help="Pipe-separated acceptance criteria")

    # task-update
    p = sub.add_parser("task-update", help="Update task status or progress")
    p.add_argument("--id", required=True)
    p.add_argument("--status", choices=["pending", "assigned", "in_progress", "blocked", "failed", "abandoned"])
    p.add_argument("--assign", help="Reassign to instance")
    p.add_argument("--progress", help="Progress summary")
    p.add_argument("--files-modified", help="Comma-separated modified files")

    # task-complete
    p = sub.add_parser("task-complete", help="Mark a task as completed")
    p.add_argument("--id", required=True)
    p.add_argument("--commit", help="Commit SHA")
    p.add_argument("--summary", required=True, help="Completion summary")

    # decide
    p = sub.add_parser("decide", help="Record a decision")
    p.add_argument("--decision", required=True)
    p.add_argument("--rationale", required=True)
    p.add_argument("--irreversible", action="store_true")

    # discover
    p = sub.add_parser("discover", help="Record a discovery")
    p.add_argument("--text", required=True)

    # context
    p = sub.add_parser("context", help="Record a file read during orientation")
    p.add_argument("--file", required=True)

    # blocker
    p = sub.add_parser("blocker", help="Record a blocker")
    p.add_argument("--desc", required=True)
    p.add_argument("--task", required=True, help="Blocked task ID")

    # status
    sub.add_parser("status", help="Show current checkpoint status")

    # handoff
    p = sub.add_parser("handoff", help="Prepare graceful handoff to successor")
    p.add_argument("--to", required=True, help="Successor session ID")
    p.add_argument("--context", help="Critical context for successor")

    # validate
    sub.add_parser("validate", help="Validate checkpoint integrity")

    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "task-add": cmd_task_add,
        "task-update": cmd_task_update,
        "task-complete": cmd_task_complete,
        "decide": cmd_decide,
        "discover": cmd_discover,
        "context": cmd_context,
        "blocker": cmd_blocker,
        "status": cmd_status,
        "handoff": cmd_handoff,
        "validate": cmd_validate,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
