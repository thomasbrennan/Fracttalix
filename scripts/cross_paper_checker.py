#!/usr/bin/env python3
"""Cross-paper consistency checker for Fracttalix AI layers.

Validates that:
1. derivation_source claim-ID references point to existing claim IDs
2. Placeholder target_paper/target_claim references are valid
3. Process graph dependencies match claim derivation chains
4. No orphan claims (claims referenced by nothing and referencing nothing)

Prose references (section numbers, first principles, etc.) in derivation_source
are recognized and skipped — only formal claim IDs are validated.

Usage:
    python scripts/cross_paper_checker.py [--json] [--check-only]
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

LAYER_DIR = Path(__file__).resolve().parent.parent / "ai-layers"

# Pattern matching formal claim IDs: A-1.1, D-MK1.2, F-SFW.3, C-DRP.1, etc.
CLAIM_ID_PATTERN = re.compile(r"^[ADFC]-[A-Za-z]*\d*\.\d+$")


def is_claim_id(ref):
    """Return True if ref looks like a formal claim ID, not a prose reference."""
    # Also handle cross-paper colon notation: "P1:F-1.4" -> "F-1.4"
    cleaned = ref.split(":")[-1] if ":" in ref else ref
    return bool(CLAIM_ID_PATTERN.match(cleaned))


def normalize_ref(ref):
    """Normalize a claim reference (strip cross-paper prefix)."""
    if ":" in ref:
        return ref.split(":")[-1]
    return ref


def load_all_layers():
    """Load all AI layer files, return dict keyed by paper_id."""
    layers = {}
    for f in sorted(LAYER_DIR.glob("*-ai-layer.json")):
        with open(f) as fh:
            data = json.load(fh)
        layers[data["paper_id"]] = {
            "file": f.name,
            "data": data,
        }
    return layers


def load_process_graph():
    pg_path = LAYER_DIR / "process-graph.json"
    if pg_path.exists():
        with open(pg_path) as f:
            return json.load(f)
    return None


def collect_all_claim_ids(layers):
    """Return set of all claim IDs across all layers."""
    ids = set()
    for pid, entry in layers.items():
        for claim in entry["data"].get("claim_registry", []):
            ids.add(claim["claim_id"])
    return ids


def check_derivation_sources(layers, all_ids):
    """Check that derivation_source claim-ID references point to existing claims.
    Prose references (section numbers, first principles, etc.) are skipped."""
    issues = []
    prose_refs = 0
    for pid, entry in layers.items():
        for claim in entry["data"].get("claim_registry", []):
            sources = claim.get("derivation_source") or []
            for src in sources:
                normalized = normalize_ref(src)
                if not is_claim_id(normalized):
                    prose_refs += 1
                    continue
                if normalized not in all_ids:
                    issues.append({
                        "type": "BROKEN_DERIVATION_SOURCE",
                        "paper": pid,
                        "claim": claim["claim_id"],
                        "reference": src,
                        "message": f"{claim['claim_id']} references claim {src} but no such claim exists in any layer"
                    })
    return issues, prose_refs


def check_placeholder_targets(layers, all_ids):
    """Check placeholder register target references."""
    issues = []
    all_paper_ids = set(layers.keys())
    for pid, entry in layers.items():
        for ph in entry["data"].get("placeholder_register", []):
            # Check source_claim exists in this layer
            local_ids = {c["claim_id"] for c in entry["data"].get("claim_registry", [])}
            if ph["source_claim"] not in local_ids:
                issues.append({
                    "type": "BROKEN_PLACEHOLDER_SOURCE",
                    "paper": pid,
                    "placeholder": ph["placeholder_id"],
                    "reference": ph["source_claim"],
                    "message": f"Placeholder {ph['placeholder_id']} references source_claim {ph['source_claim']} not found in {pid}"
                })
            # Check target_paper exists (warning only — paper may not have layer yet)
            tp = ph.get("target_paper")
            if tp and tp not in all_paper_ids:
                issues.append({
                    "type": "MISSING_TARGET_PAPER_LAYER",
                    "paper": pid,
                    "placeholder": ph["placeholder_id"],
                    "reference": tp,
                    "message": f"Placeholder {ph['placeholder_id']} targets paper {tp} which has no AI layer",
                    "severity": "warning"
                })
            # Check target_claim if specified (warning — target may not exist yet)
            tc = ph.get("target_claim")
            if tc and tc is not None and is_claim_id(tc) and tc not in all_ids:
                issues.append({
                    "type": "UNRESOLVED_PLACEHOLDER_TARGET",
                    "paper": pid,
                    "placeholder": ph["placeholder_id"],
                    "reference": tc,
                    "message": f"Placeholder {ph['placeholder_id']} targets claim {tc} not yet in any layer",
                    "severity": "warning"
                })
    return issues


def check_dependency_coverage(layers, process_graph):
    """Check that process graph dependencies are reflected in derivation chains."""
    issues = []
    if not process_graph:
        return issues

    # For each paper, what other papers does it reference via derivation_source?
    paper_refs = defaultdict(set)
    for pid, entry in layers.items():
        for claim in entry["data"].get("claim_registry", []):
            for src in (claim.get("derivation_source") or []):
                normalized = normalize_ref(src)
                if not is_claim_id(normalized):
                    continue
                parts = normalized.split("-", 1)
                if len(parts) == 2:
                    ref = parts[1].split(".")[0]
                    # Map claim ref back to paper ID
                    if ref.startswith("MK"):
                        ref_paper = f"MK-P{ref[2:]}"
                    elif ref.startswith("DRP"):
                        ref_paper = f"DRP-{ref[3:]}" if len(ref) > 3 else "DRP-1"
                    elif ref.startswith("SFW"):
                        ref_paper = f"SFW-{ref[3:]}" if len(ref) > 3 else "SFW-1"
                    else:
                        try:
                            ref_paper = f"P{int(ref)}"
                        except ValueError:
                            continue
                    if ref_paper != pid:
                        paper_refs[pid].add(ref_paper)

    for corpus in process_graph.get("corpora", []):
        for dep in corpus.get("dependencies", []):
            from_paper = dep["from"]
            to_paper = dep["to"]
            if to_paper in layers and layers[to_paper]["data"].get("claim_registry"):
                claims = layers[to_paper]["data"]["claim_registry"]
                if claims and from_paper not in paper_refs.get(to_paper, set()):
                    issues.append({
                        "type": "DEPENDENCY_NOT_REFLECTED",
                        "paper": to_paper,
                        "dependency": from_paper,
                        "message": f"Process graph: {to_paper} depends on {from_paper}, but no claims in {to_paper} reference {from_paper} claims",
                        "severity": "warning"
                    })

    return issues


def check_orphan_claims(layers):
    """Find non-axiom claims that are neither referenced by nor reference anything."""
    issues = []
    all_referenced = set()

    for pid, entry in layers.items():
        for claim in entry["data"].get("claim_registry", []):
            for src in (claim.get("derivation_source") or []):
                all_referenced.add(normalize_ref(src))

    for pid, entry in layers.items():
        for claim in entry["data"].get("claim_registry", []):
            cid = claim["claim_id"]
            has_sources = bool(claim.get("derivation_source"))
            is_referenced = cid in all_referenced
            if not has_sources and not is_referenced and claim["type"] != "A":
                issues.append({
                    "type": "ORPHAN_CLAIM",
                    "paper": pid,
                    "claim": cid,
                    "message": f"{cid} ({claim.get('name', '?')}) — no derivation sources and not referenced by any other claim",
                    "severity": "warning"
                })

    return issues


def main():
    json_mode = "--json" in sys.argv
    check_only = "--check-only" in sys.argv

    layers = load_all_layers()
    pg = load_process_graph()
    all_ids = collect_all_claim_ids(layers)

    all_issues = []
    deriv_issues, prose_refs = check_derivation_sources(layers, all_ids)
    all_issues.extend(deriv_issues)
    all_issues.extend(check_placeholder_targets(layers, all_ids))
    all_issues.extend(check_dependency_coverage(layers, pg))
    all_issues.extend(check_orphan_claims(layers))

    errors = [i for i in all_issues if i.get("severity") != "warning"]
    warnings = [i for i in all_issues if i.get("severity") == "warning"]

    if json_mode:
        result = {
            "layers_checked": len(layers),
            "total_claims": len(all_ids),
            "prose_references_skipped": prose_refs,
            "errors": len(errors),
            "warnings": len(warnings),
            "issues": all_issues,
        }
        print(json.dumps(result, indent=2))
        sys.exit(1 if errors else 0)

    if not check_only:
        print("Fracttalix Cross-Paper Consistency Checker")
        print(f"Layers: {len(layers)} | Claims: {len(all_ids)} | Prose refs skipped: {prose_refs}")
        print("=" * 60)

    if errors:
        if not check_only:
            print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  [{e['type']}] {e['message']}")

    if warnings and not check_only:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  [{w['type']}] {w['message']}")

    if not errors and not warnings:
        if not check_only:
            print("\nAll cross-references valid. No issues found.")
    elif not errors:
        if not check_only:
            print(f"\n{len(warnings)} warnings, 0 errors. Cross-references valid.")

    if not check_only:
        # Summary
        refs = 0
        claim_refs = 0
        for pid, entry in layers.items():
            for claim in entry["data"].get("claim_registry", []):
                sources = claim.get("derivation_source") or []
                refs += len(sources)
                claim_refs += sum(1 for s in sources if is_claim_id(normalize_ref(s)))
        print(f"\nCross-reference summary:")
        print(f"  Total derivation_source entries: {refs} ({claim_refs} claim IDs, {prose_refs} prose refs)")
        print(f"  Unique claim IDs: {len(all_ids)}")
        print(f"  Papers with claims: {sum(1 for e in layers.values() if e['data'].get('claim_registry'))}")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
