#!/usr/bin/env python3
"""DDN Health Monitor — continuous network health monitoring with self-healing.

Designed to run as a cron job or systemd timer. Checks all network nodes,
logs status, and triggers automatic recovery when nodes fall out of sync.

Usage:
    python -m network.health_monitor              # One-shot health check
    python -m network.health_monitor --auto-heal  # Check + auto-recover
    python -m network.health_monitor --json       # Machine-readable output

Cron example (every 6 hours):
    0 */6 * * * cd /path/to/Fracttalix && python -m network.health_monitor --auto-heal >> /var/log/ddn-health.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from network.distributed_docs import (
    EXIT_CRITICAL,
    EXIT_DEGRADED,
    EXIT_OK,
    REPO_ROOT,
    check_git_remote_health,
    check_http_health,
    get_configured_remotes,
    get_local_head,
    load_manifest,
    load_state,
    push_with_retry,
    save_state,
    _remote_name_for_node,
)

HEALTH_LOG_PATH = Path(__file__).parent / ".health-log.jsonl"


def check_all_nodes() -> dict:
    """Check every node in the manifest and return structured results."""
    manifest = load_manifest()
    configured = get_configured_remotes()
    local_head = get_local_head()
    now = datetime.now(timezone.utc).isoformat()

    results = {
        "timestamp": now,
        "local_head": local_head,
        "nodes": [],
        "reachable": 0,
        "total": 0,
        "diverged": [],
        "unreachable": [],
    }

    for node in manifest["nodes"]:
        if not node["enabled"]:
            continue

        results["total"] += 1
        entry = {
            "id": node["id"],
            "provider": node["provider"],
            "type": node["type"],
            "reachable": False,
            "in_sync": None,
            "error": None,
        }

        if node["type"] == "git-remote":
            remote_name = _remote_name_for_node(node)
            if remote_name in configured:
                ok, head, err = check_git_remote_health(remote_name)
                entry["reachable"] = ok
                entry["error"] = err
                if ok and head:
                    entry["in_sync"] = head == local_head
                    entry["remote_head"] = head
                    if not entry["in_sync"]:
                        results["diverged"].append(node["id"])
            else:
                endpoint = node.get("health_endpoint")
                if endpoint:
                    ok, _, err = check_http_health(endpoint)
                    entry["reachable"] = ok
                    entry["error"] = err
                else:
                    entry["error"] = "Remote not configured"

        elif node["type"] in ("archive", "content-addressed"):
            url = node.get("health_endpoint") or node.get("url")
            if url:
                ok, latency, err = check_http_health(url)
                entry["reachable"] = ok
                entry["error"] = err
                entry["latency_ms"] = latency

        if entry["reachable"]:
            results["reachable"] += 1
        else:
            results["unreachable"].append(node["id"])

        results["nodes"].append(entry)

    return results


def auto_heal(results: dict) -> list[str]:
    """Attempt to heal diverged nodes by pushing current state."""
    manifest = load_manifest()
    configured = get_configured_remotes()
    healed = []

    for node_id in results["diverged"]:
        node = next((n for n in manifest["nodes"] if n["id"] == node_id), None)
        if not node or node["type"] != "git-remote":
            continue

        remote_name = _remote_name_for_node(node)
        if remote_name not in configured:
            continue

        print(f"  Auto-healing {node_id}...")
        ok_branches = push_with_retry(remote_name, "--all")
        ok_tags = push_with_retry(remote_name, "--tags")

        if ok_branches and ok_tags:
            print(f"    HEALED — {node_id} synced successfully")
            healed.append(node_id)
        else:
            print(f"    FAILED — {node_id} could not be synced")

    return healed


def append_health_log(results: dict) -> None:
    """Append a health check result to the JSONL log."""
    summary = {
        "timestamp": results["timestamp"],
        "reachable": results["reachable"],
        "total": results["total"],
        "diverged": results["diverged"],
        "unreachable": results["unreachable"],
    }
    with open(HEALTH_LOG_PATH, "a") as f:
        f.write(json.dumps(summary) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="health_monitor",
        description="DDN Health Monitor",
    )
    parser.add_argument("--auto-heal", action="store_true",
                        help="Automatically attempt to heal diverged nodes")
    parser.add_argument("--json", action="store_true",
                        help="Output machine-readable JSON")
    args = parser.parse_args()

    results = check_all_nodes()
    append_health_log(results)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        manifest = load_manifest()
        quorum = manifest["network"]["quorum_threshold"]

        print(f"[{results['timestamp']}] DDN Health Check")
        print(f"  Nodes: {results['reachable']}/{results['total']} reachable")
        if results["diverged"]:
            print(f"  Diverged: {', '.join(results['diverged'])}")
        if results["unreachable"]:
            print(f"  Unreachable: {', '.join(results['unreachable'])}")

        if results["reachable"] >= quorum:
            print(f"  Quorum: MET ({results['reachable']}/{quorum})")
        else:
            print(f"  Quorum: NOT MET ({results['reachable']}/{quorum})")

    # Auto-heal if requested
    if args.auto_heal and results["diverged"]:
        print()
        print("Initiating auto-heal...")
        healed = auto_heal(results)
        if healed:
            print(f"  Healed {len(healed)} node(s)")

    # Update state
    state = load_state()
    state["last_health_check"] = results["timestamp"]
    state["node_states"] = {
        n["id"]: {"reachable": n["reachable"], "in_sync": n.get("in_sync")}
        for n in results["nodes"]
    }
    save_state(state)

    # Exit code reflects network health
    if results["reachable"] >= load_manifest()["network"]["minimum_replicas"]:
        return EXIT_OK
    elif results["reachable"] >= load_manifest()["network"]["quorum_threshold"]:
        return EXIT_DEGRADED
    else:
        return EXIT_CRITICAL


if __name__ == "__main__":
    sys.exit(main())
