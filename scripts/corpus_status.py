#!/usr/bin/env python3
"""Fracttalix Corpus Status Report Generator.

Reads all AI layers and the process graph to produce a machine-readable
and human-readable status report of the corpus.

Usage:
  python scripts/corpus_status.py              # Full report
  python scripts/corpus_status.py --check-only # Exit 1 if integrity issues found
  python scripts/corpus_status.py --json       # JSON output
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AI_LAYERS_DIR = REPO_ROOT / "ai-layers"
PROCESS_GRAPH_PATH = AI_LAYERS_DIR / "process-graph.json"


def load_layers():
    layers = {}
    for filepath in sorted(AI_LAYERS_DIR.glob("*-ai-layer.json")):
        with open(filepath) as f:
            data = json.load(f)
        layers[data.get("paper_id", filepath.stem)] = {
            "file": filepath.name,
            "paper_id": data.get("paper_id"),
            "paper_title": data.get("paper_title"),
            "paper_type": data.get("paper_type"),
            "version": data.get("version"),
            "session": data.get("session"),
            "phase_ready": data.get("phase_ready", {}).get("verdict"),
            "total_claims": len(data.get("claim_registry", [])),
            "type_A": sum(1 for c in data.get("claim_registry", []) if c.get("type") == "A"),
            "type_D": sum(1 for c in data.get("claim_registry", []) if c.get("type") == "D"),
            "type_F": sum(1 for c in data.get("claim_registry", []) if c.get("type") == "F"),
            "placeholders_total": len(data.get("placeholder_register", [])),
            "placeholders_unresolved": sum(
                1 for p in data.get("placeholder_register", []) if not p.get("resolved", True)
            ),
            "claim_ids": [c.get("claim_id") for c in data.get("claim_registry", [])],
        }
    return layers


def load_process_graph():
    if not PROCESS_GRAPH_PATH.exists():
        return None
    with open(PROCESS_GRAPH_PATH) as f:
        return json.load(f)


def check_integrity(layers, process_graph):
    """Return list of integrity issues."""
    issues = []

    # Cross-layer claim reference check
    all_claim_ids = {}
    for pid, layer in layers.items():
        for cid in layer["claim_ids"]:
            all_claim_ids[cid] = pid

    # Check process graph nodes have corresponding AI layers
    if process_graph:
        for corpus in process_graph.get("corpora", []):
            for paper in corpus.get("papers", []):
                pid = paper.get("paper_id", "")
                status = paper.get("status", "")
                if "PHASE-READY" in status:
                    if pid not in layers:
                        issues.append(
                            f"Process graph paper '{pid}' ({status}) has no AI layer file"
                        )

    return issues


def print_report(layers, process_graph, issues):
    print("=" * 70)
    print("FRACTTALIX CORPUS STATUS REPORT")
    print("=" * 70)

    # Summary
    total_claims = sum(l["total_claims"] for l in layers.values())
    total_f = sum(l["type_F"] for l in layers.values())
    total_d = sum(l["type_D"] for l in layers.values())
    total_a = sum(l["type_A"] for l in layers.values())
    total_ph = sum(l["placeholders_unresolved"] for l in layers.values())
    phase_ready_count = sum(1 for l in layers.values() if l["phase_ready"] == "PHASE-READY")

    print(f"\nAI Layers:          {len(layers)}")
    print(f"Phase-Ready:        {phase_ready_count}/{len(layers)}")
    print(f"Total Claims:       {total_claims} (A:{total_a} D:{total_d} F:{total_f})")
    print(f"Open Placeholders:  {total_ph}")
    print(f"Integrity Issues:   {len(issues)}")

    # Per-layer detail
    print("\n" + "-" * 70)
    print(f"{'Paper ID':<12} {'Version':<8} {'Session':<8} {'Status':<16} {'Claims':<8} {'PH':<4}")
    print("-" * 70)
    for pid, layer in layers.items():
        pid = layer['paper_id'] or "?"
        ver = layer['version'] or "?"
        sess = layer['session'] or "?"
        pr = layer['phase_ready'] or "?"
        print(f"{pid:<12} {ver:<8} {sess:<8} {pr:<16} {layer['total_claims']:<8} "
              f"{layer['placeholders_unresolved']:<4}")

    # Process graph status
    if process_graph:
        for corpus in process_graph.get("corpora", []):
            cid = corpus.get("corpus_id", "?")
            papers = corpus.get("papers", [])
            deps = corpus.get("dependencies", [])

            print(f"\n{'-' * 70}")
            print(f"PROCESS GRAPH — {cid} ({len(papers)} papers)")
            print("-" * 70)
            for paper in papers:
                pid = paper.get("paper_id", "")
                title = paper.get("title", "")
                status = paper.get("status", "")
                has_layer = "Y" if pid in layers else "-"
                sync = " [SYNC]" if paper.get("sync_point") else ""
                risk = f" [RISK:{paper.get('risk_level')}]" if paper.get("risk_level") else ""
                print(f"  {pid:<8} [{has_layer}] {status:<20} {title}{sync}{risk}")

            if deps:
                print(f"\n  Dependencies ({len(deps)} edges):")
                for edge in deps:
                    print(f"    {edge['from']} -> {edge['to']}")

    # Integrity issues
    if issues:
        print("\n" + "-" * 70)
        print("INTEGRITY ISSUES")
        print("-" * 70)
        for issue in issues:
            print(f"  ! {issue}")

    print("\n" + "=" * 70)


def main():
    check_only = "--check-only" in sys.argv
    json_output = "--json" in sys.argv

    layers = load_layers()
    process_graph = load_process_graph()
    issues = check_integrity(layers, process_graph)

    if json_output:
        report = {
            "layers": {k: {kk: vv for kk, vv in v.items() if kk != "claim_ids"}
                       for k, v in layers.items()},
            "total_claims": sum(l["total_claims"] for l in layers.values()),
            "phase_ready_count": sum(1 for l in layers.values()
                                     if l["phase_ready"] == "PHASE-READY"),
            "open_placeholders": sum(l["placeholders_unresolved"] for l in layers.values()),
            "integrity_issues": issues,
        }
        print(json.dumps(report, indent=2))
    elif check_only:
        if issues:
            for issue in issues:
                print(f"ISSUE: {issue}")
            sys.exit(1)
        else:
            print("All integrity checks passed.")
            sys.exit(0)
    else:
        print_report(layers, process_graph, issues)

    if issues and not json_output:
        sys.exit(1)


if __name__ == "__main__":
    main()
