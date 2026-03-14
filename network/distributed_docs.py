#!/usr/bin/env python3
"""Fracttalix Distributed Docs Network (DDN) Manager.

A network persists; a node invites loss.

This tool manages the distributed documentation network for Fracttalix,
ensuring that all documentation, AI layers, and research artifacts are
replicated across multiple independent providers. If any single provider
goes down, the network routes around the damage.

Usage:
    python -m network.distributed_docs status        # Show network health
    python -m network.distributed_docs sync           # Sync all mirrors
    python -m network.distributed_docs snapshot       # Create IPFS snapshot
    python -m network.distributed_docs recover <id>   # Recover a failed node
    python -m network.distributed_docs verify         # Verify integrity across nodes
    python -m network.distributed_docs add-remote     # Add git mirrors to local repo
    python -m network.distributed_docs bootstrap      # First-time network setup
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

MANIFEST_PATH = Path(__file__).parent / "manifest.json"
REPO_ROOT = Path(__file__).parent.parent
NETWORK_STATE_PATH = Path(__file__).parent / ".network-state.json"

# Exit codes
EXIT_OK = 0
EXIT_DEGRADED = 1
EXIT_CRITICAL = 2


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NodeStatus:
    node_id: str
    provider: str
    role: str
    reachable: bool
    head_commit: Optional[str] = None
    last_checked: Optional[str] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None


@dataclass
class NetworkHealth:
    timestamp: str
    total_nodes: int
    reachable_nodes: int
    git_mirrors_in_sync: bool
    head_commit: Optional[str] = None
    node_statuses: list[NodeStatus] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return self.reachable_nodes >= self._quorum

    @property
    def _quorum(self) -> int:
        return load_manifest()["network"]["quorum_threshold"]


# ---------------------------------------------------------------------------
# Manifest and state management
# ---------------------------------------------------------------------------

def load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def load_state() -> dict:
    if NETWORK_STATE_PATH.exists():
        with open(NETWORK_STATE_PATH) as f:
            return json.load(f)
    return {"last_sync": None, "last_health_check": None, "node_states": {}}


def save_state(state: dict) -> None:
    with open(NETWORK_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Git operations
# ---------------------------------------------------------------------------

def git(*args: str, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def get_local_head() -> Optional[str]:
    result = git("rev-parse", "HEAD")
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def get_local_branches() -> list[str]:
    result = git("branch", "-a", "--format=%(refname:short)")
    if result.returncode == 0:
        return [b.strip() for b in result.stdout.strip().split("\n") if b.strip()]
    return []


def get_local_tags() -> list[str]:
    result = git("tag", "-l")
    if result.returncode == 0:
        return [t.strip() for t in result.stdout.strip().split("\n") if t.strip()]
    return []


def get_remote_head(remote_name: str) -> Optional[str]:
    result = git("ls-remote", remote_name, "HEAD")
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().split()[0]
    return None


def push_with_retry(remote: str, refspec: str = "--all",
                    max_attempts: int = 4, base_delay: int = 2) -> bool:
    """Push to remote with exponential backoff retry."""
    for attempt in range(max_attempts):
        result = git("push", remote, refspec)
        if result.returncode == 0:
            return True
        if attempt < max_attempts - 1:
            delay = base_delay * (2 ** attempt)
            print(f"  Push to {remote} failed (attempt {attempt + 1}/{max_attempts}), "
                  f"retrying in {delay}s...")
            time.sleep(delay)
    return False


# ---------------------------------------------------------------------------
# Network health checks
# ---------------------------------------------------------------------------

def check_http_health(url: str, timeout: int = 10) -> tuple[bool, float, Optional[str]]:
    """Check if an HTTP endpoint is reachable. Returns (ok, latency_ms, error)."""
    start = time.monotonic()
    try:
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "Fracttalix-DDN/1.0")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            latency = (time.monotonic() - start) * 1000
            if resp.status < 400:
                return True, latency, None
            return False, latency, f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        latency = (time.monotonic() - start) * 1000
        # Some APIs return 404 for non-existent repos but the service is up
        return False, latency, f"HTTP {e.code}"
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        return False, latency, str(e)


def check_git_remote_health(remote_name: str) -> tuple[bool, Optional[str], Optional[str]]:
    """Check a git remote. Returns (reachable, head_commit, error)."""
    try:
        result = git("ls-remote", remote_name, "HEAD")
        if result.returncode == 0 and result.stdout.strip():
            head = result.stdout.strip().split()[0]
            return True, head, None
        return False, None, result.stderr.strip() or "No HEAD ref found"
    except subprocess.TimeoutExpired:
        return False, None, "Timeout"
    except Exception as e:
        return False, None, str(e)


def get_configured_remotes() -> dict[str, str]:
    """Get all configured git remotes as {name: url}."""
    result = git("remote", "-v")
    remotes = {}
    if result.returncode == 0:
        for line in result.stdout.strip().split("\n"):
            if line and "(push)" in line:
                parts = line.split()
                remotes[parts[0]] = parts[1]
    return remotes


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_status() -> int:
    """Show the current health of the distributed network."""
    manifest = load_manifest()
    local_head = get_local_head()
    configured_remotes = get_configured_remotes()
    now = datetime.now(timezone.utc).isoformat()

    print("=" * 70)
    print("  FRACTTALIX DISTRIBUTED DOCS NETWORK — STATUS")
    print(f"  {now}")
    print("=" * 70)
    print()
    print(f"  Local HEAD: {local_head or 'unknown'}")
    print(f"  Configured remotes: {', '.join(configured_remotes.keys()) or 'none'}")
    print()

    reachable = 0
    total_git = 0
    in_sync = 0
    statuses: list[NodeStatus] = []

    for node in manifest["nodes"]:
        if not node["enabled"]:
            continue

        node_id = node["id"]
        provider = node["provider"]
        role = node["role"]

        if node["type"] == "git-remote":
            total_git += 1
            # Check if we have this remote configured
            remote_name = _remote_name_for_node(node)
            if remote_name in configured_remotes:
                ok, head, err = check_git_remote_health(remote_name)
                latency = None
            else:
                # Try health endpoint instead
                endpoint = node.get("health_endpoint")
                if endpoint:
                    ok, latency, err = check_http_health(endpoint)
                    head = None
                else:
                    ok, latency, err = False, None, "Not configured"
                    head = None

            status = NodeStatus(
                node_id=node_id, provider=provider, role=role,
                reachable=ok, head_commit=head, last_checked=now,
                error=err, latency_ms=latency,
            )
            if ok:
                reachable += 1
                if head and head == local_head:
                    in_sync += 1

        elif node["type"] in ("archive", "content-addressed"):
            endpoint = node.get("health_endpoint") or node.get("url")
            if endpoint:
                ok, latency, err = check_http_health(endpoint)
            else:
                ok, latency, err = False, None, "No endpoint"

            status = NodeStatus(
                node_id=node_id, provider=provider, role=role,
                reachable=ok, last_checked=now, error=err,
                latency_ms=latency,
            )
            if ok:
                reachable += 1
        else:
            continue

        statuses.append(status)

    # Print results
    quorum = manifest["network"]["quorum_threshold"]
    min_replicas = manifest["network"]["minimum_replicas"]

    print("  NODE STATUS")
    print("  " + "-" * 66)
    for s in statuses:
        icon = "[OK]" if s.reachable else "[!!]"
        sync_info = ""
        if s.head_commit:
            if s.head_commit == local_head:
                sync_info = " (in sync)"
            else:
                sync_info = f" (DIVERGED: {s.head_commit[:8]})"
        latency_info = f" {s.latency_ms:.0f}ms" if s.latency_ms else ""
        error_info = f" — {s.error}" if s.error and not s.reachable else ""
        print(f"  {icon} {s.node_id:<25} {s.provider:<18} {s.role:<12}"
              f"{sync_info}{latency_info}{error_info}")
    print()

    # Summary
    total = len(statuses)
    print(f"  Reachable: {reachable}/{total} nodes")
    if total_git > 0:
        print(f"  Git mirrors in sync: {in_sync}/{total_git}")
    print(f"  Quorum: {'MET' if reachable >= quorum else 'NOT MET'} "
          f"({reachable}/{quorum} required)")
    print(f"  Min replicas: {'MET' if reachable >= min_replicas else 'NOT MET'} "
          f"({reachable}/{min_replicas} required)")
    print()

    if reachable >= min_replicas:
        print("  NETWORK HEALTHY")
        return EXIT_OK
    elif reachable >= quorum:
        print("  NETWORK DEGRADED — add more replicas")
        return EXIT_DEGRADED
    else:
        print("  NETWORK CRITICAL — below quorum threshold")
        return EXIT_CRITICAL


def cmd_sync() -> int:
    """Sync all configured git mirrors."""
    manifest = load_manifest()
    configured_remotes = get_configured_remotes()
    policy = manifest["sync_policy"]["on_push_to_main"]
    success_count = 0
    fail_count = 0

    print("Syncing distributed documentation network...")
    print()

    for node in manifest["nodes"]:
        if not node["enabled"] or node["type"] != "git-remote":
            continue
        if node["role"] == "primary":
            continue

        remote_name = _remote_name_for_node(node)
        if remote_name not in configured_remotes:
            print(f"  SKIP {node['id']} — remote '{remote_name}' not configured. "
                  f"Run 'add-remote' first.")
            fail_count += 1
            continue

        print(f"  Syncing {node['id']} ({node['provider']})...")

        # Push all branches
        retry = policy["retry"]
        ok_branches = push_with_retry(
            remote_name, "--all",
            max_attempts=retry["max_attempts"],
            base_delay=retry["base_delay_seconds"],
        )

        # Push all tags
        ok_tags = push_with_retry(
            remote_name, "--tags",
            max_attempts=retry["max_attempts"],
            base_delay=retry["base_delay_seconds"],
        )

        if ok_branches and ok_tags:
            print(f"    OK — all branches and tags pushed")
            success_count += 1
        else:
            what_failed = []
            if not ok_branches:
                what_failed.append("branches")
            if not ok_tags:
                what_failed.append("tags")
            print(f"    FAILED — could not push: {', '.join(what_failed)}")
            fail_count += 1

    print()
    print(f"Sync complete: {success_count} succeeded, {fail_count} failed")

    # Update state
    state = load_state()
    state["last_sync"] = datetime.now(timezone.utc).isoformat()
    save_state(state)

    return EXIT_OK if fail_count == 0 else EXIT_DEGRADED


def cmd_add_remote() -> int:
    """Add all network mirror remotes to the local git config."""
    manifest = load_manifest()
    configured = get_configured_remotes()
    added = 0

    print("Adding network remotes to local git configuration...")
    print()

    for node in manifest["nodes"]:
        if not node["enabled"] or node["type"] != "git-remote":
            continue

        remote_name = _remote_name_for_node(node)
        url = node["url"]

        if remote_name in configured:
            if configured[remote_name] == url:
                print(f"  EXISTS {remote_name} -> {url}")
            else:
                print(f"  UPDATE {remote_name}: {configured[remote_name]} -> {url}")
                git("remote", "set-url", remote_name, url)
                added += 1
        else:
            print(f"  ADD    {remote_name} -> {url}")
            git("remote", "add", remote_name, url)
            added += 1

    print()
    print(f"Done. {added} remote(s) added/updated.")
    return EXIT_OK


def cmd_verify() -> int:
    """Verify integrity across all reachable git mirrors."""
    manifest = load_manifest()
    local_head = get_local_head()
    local_branches = set(get_local_branches())
    local_tags = set(get_local_tags())
    configured = get_configured_remotes()
    all_ok = True

    print("Verifying network integrity...")
    print(f"  Local HEAD: {local_head}")
    print(f"  Local branches: {len(local_branches)}")
    print(f"  Local tags: {len(local_tags)}")
    print()

    for node in manifest["nodes"]:
        if not node["enabled"] or node["type"] != "git-remote":
            continue

        remote_name = _remote_name_for_node(node)
        if remote_name not in configured:
            continue

        print(f"  Checking {node['id']}...")

        # HEAD comparison
        ok, remote_head, err = check_git_remote_health(remote_name)
        if not ok:
            print(f"    UNREACHABLE: {err}")
            all_ok = False
            continue

        if remote_head == local_head:
            print(f"    HEAD: OK (matches local)")
        else:
            print(f"    HEAD: DIVERGED — remote={remote_head[:12]}, local={local_head[:12]}")
            all_ok = False

        # Branch count
        result = git("ls-remote", "--heads", remote_name)
        if result.returncode == 0:
            remote_branch_count = len([
                l for l in result.stdout.strip().split("\n") if l.strip()
            ])
            print(f"    Branches: {remote_branch_count} remote "
                  f"(local has {len(local_branches)})")

        # Tag count
        result = git("ls-remote", "--tags", remote_name)
        if result.returncode == 0:
            remote_tag_count = len([
                l for l in result.stdout.strip().split("\n")
                if l.strip() and "^{}" not in l
            ])
            print(f"    Tags: {remote_tag_count} remote (local has {len(local_tags)})")

    print()
    if all_ok:
        print("  INTEGRITY VERIFIED — all reachable nodes consistent")
        return EXIT_OK
    else:
        print("  INTEGRITY ISSUES — run 'sync' to reconcile")
        return EXIT_DEGRADED


def cmd_snapshot() -> int:
    """Create a documentation snapshot for content-addressed storage (IPFS)."""
    import hashlib as _hashlib
    import tarfile
    import tempfile

    manifest = load_manifest()
    include_patterns = manifest["sync_policy"]["on_release"]["include_patterns"]

    print("Creating documentation snapshot for distributed archival...")
    print()

    # Collect all documentation files
    doc_files: list[Path] = []
    for pattern in include_patterns:
        from glob import glob as _glob
        matches = _glob(str(REPO_ROOT / pattern), recursive=True)
        doc_files.extend(Path(m) for m in matches if Path(m).is_file())

    # Deduplicate and sort
    doc_files = sorted(set(doc_files))
    print(f"  Collected {len(doc_files)} documentation files")

    # Create tarball
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    snapshot_name = f"fracttalix-docs-{timestamp}"
    snapshot_dir = REPO_ROOT / "network" / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    tarball_path = snapshot_dir / f"{snapshot_name}.tar.gz"

    with tarfile.open(tarball_path, "w:gz") as tar:
        for fp in doc_files:
            arcname = fp.relative_to(REPO_ROOT)
            tar.add(fp, arcname=str(arcname))

    # Compute content hash (serves as a pre-CID integrity check)
    sha256 = _hashlib.sha256()
    with open(tarball_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    content_hash = sha256.hexdigest()

    size_kb = tarball_path.stat().st_size / 1024
    print(f"  Snapshot: {tarball_path.name}")
    print(f"  Size: {size_kb:.1f} KB")
    print(f"  SHA-256: {content_hash}")
    print(f"  Files: {len(doc_files)}")
    print()

    # Write snapshot manifest
    snapshot_manifest = {
        "name": snapshot_name,
        "created": datetime.now(timezone.utc).isoformat(),
        "sha256": content_hash,
        "file_count": len(doc_files),
        "size_bytes": tarball_path.stat().st_size,
        "git_head": get_local_head(),
        "archive_path": str(tarball_path.relative_to(REPO_ROOT)),
        "ipfs_cid": None,
        "pin_status": {},
    }

    manifest_out = snapshot_dir / f"{snapshot_name}.manifest.json"
    with open(manifest_out, "w") as f:
        json.dump(snapshot_manifest, f, indent=2)

    print(f"  Manifest: {manifest_out.name}")
    print()
    print("  To pin to IPFS:")
    print(f"    ipfs add -r {tarball_path}")
    print("    # or via pinning service:")
    print(f"    curl -X POST 'https://api.pinata.cloud/pinning/pinFileToIPFS' \\")
    print(f"      -H 'Authorization: Bearer $PINATA_JWT' \\")
    print(f"      -F 'file=@{tarball_path}'")
    print()
    print("  To deposit to Internet Archive:")
    print(f"    ia upload fracttalix-docs-{timestamp} {tarball_path} \\")
    print(f"      --metadata='collection:opensource_media' \\")
    print(f"      --metadata='title:Fracttalix Documentation Snapshot {timestamp}'")

    return EXIT_OK


def cmd_recover(node_id: str) -> int:
    """Recover a failed node from surviving nodes."""
    manifest = load_manifest()
    fallback_order = manifest["recovery"]["fallback_order"]
    configured = get_configured_remotes()

    target_node = None
    for node in manifest["nodes"]:
        if node["id"] == node_id:
            target_node = node
            break

    if not target_node:
        print(f"ERROR: Unknown node '{node_id}'")
        print(f"Available nodes: {', '.join(n['id'] for n in manifest['nodes'])}")
        return EXIT_CRITICAL

    if target_node["type"] != "git-remote":
        print(f"ERROR: Recovery only supported for git-remote nodes, "
              f"not '{target_node['type']}'")
        return EXIT_CRITICAL

    print(f"Recovering node: {node_id} ({target_node['provider']})")
    print()

    # Find a healthy source
    source_remote = None
    for source_id in fallback_order:
        if source_id == node_id:
            continue
        source_node = next((n for n in manifest["nodes"] if n["id"] == source_id), None)
        if not source_node or source_node["type"] != "git-remote":
            continue
        remote_name = _remote_name_for_node(source_node)
        if remote_name in configured:
            ok, _, _ = check_git_remote_health(remote_name)
            if ok:
                source_remote = remote_name
                print(f"  Source: {source_id} ({remote_name}) — healthy")
                break

    if not source_remote:
        # Fall back to local repo
        print("  Source: local repository (no healthy remote found)")
        source_remote = None

    # Push to the target
    target_remote = _remote_name_for_node(target_node)
    if target_remote not in configured:
        print(f"  Adding remote '{target_remote}'...")
        git("remote", "add", target_remote, target_node["url"])

    # If we have a source remote, fetch from it first to ensure we're up to date
    if source_remote:
        print(f"  Fetching latest from {source_remote}...")
        git("fetch", source_remote)

    print(f"  Pushing to {target_remote}...")
    ok = push_with_retry(target_remote, "--all")
    if ok:
        push_with_retry(target_remote, "--tags")
        print()
        print(f"  RECOVERED — {node_id} is back in the network")
        return EXIT_OK
    else:
        print()
        print(f"  FAILED — could not reach {node_id}. Check provider status.")
        return EXIT_CRITICAL


def cmd_bootstrap() -> int:
    """First-time setup: configure all remotes and perform initial sync."""
    print("=" * 70)
    print("  FRACTTALIX DISTRIBUTED DOCS NETWORK — BOOTSTRAP")
    print("=" * 70)
    print()
    print("  This will configure your local repository to participate in")
    print("  the distributed documentation network.")
    print()

    # Step 1: Add remotes
    print("Step 1: Adding git remotes...")
    cmd_add_remote()
    print()

    # Step 2: Initial sync
    print("Step 2: Initial sync to all mirrors...")
    cmd_sync()
    print()

    # Step 3: Create initial snapshot
    print("Step 3: Creating initial documentation snapshot...")
    cmd_snapshot()
    print()

    # Step 4: Verify
    print("Step 4: Verifying network integrity...")
    cmd_verify()
    print()

    print("=" * 70)
    print("  BOOTSTRAP COMPLETE")
    print()
    print("  Your documentation is now configured for distributed storage.")
    print("  Run 'python -m network.distributed_docs status' to check health.")
    print()
    print("  Next steps:")
    print("  1. Create accounts on GitLab, Codeberg, Bitbucket (if needed)")
    print("  2. Create empty repos named 'Fracttalix' on each provider")
    print("  3. Run 'python -m network.distributed_docs sync' to push")
    print("  4. Set up IPFS pinning (optional, for immutable archival)")
    print("  5. The CI/CD pipeline will handle ongoing sync automatically")
    print("=" * 70)

    return EXIT_OK


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _remote_name_for_node(node: dict) -> str:
    """Derive a git remote name from a node definition."""
    provider = node["provider"]
    role = node["role"]
    if role == "primary":
        return "origin"
    return f"ddn-{provider}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="distributed_docs",
        description="Fracttalix Distributed Docs Network manager",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show network health")
    sub.add_parser("sync", help="Sync all mirrors")
    sub.add_parser("snapshot", help="Create documentation snapshot")
    sub.add_parser("verify", help="Verify integrity across nodes")
    sub.add_parser("add-remote", help="Add git remotes for all mirrors")
    sub.add_parser("bootstrap", help="First-time network setup")

    recover_p = sub.add_parser("recover", help="Recover a failed node")
    recover_p.add_argument("node_id", help="ID of the node to recover")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return EXIT_OK

    commands = {
        "status": cmd_status,
        "sync": cmd_sync,
        "snapshot": cmd_snapshot,
        "verify": cmd_verify,
        "add-remote": cmd_add_remote,
        "bootstrap": cmd_bootstrap,
    }

    if args.command == "recover":
        return cmd_recover(args.node_id)

    return commands[args.command]()


if __name__ == "__main__":
    sys.exit(main())
