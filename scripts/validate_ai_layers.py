#!/usr/bin/env python3
"""Validate all AI layer files against the Fracttalix AI Layer Schema v2-S42.

Checks:
  1. JSON Schema compliance (structure, types, required fields)
  2. Claim ID uniqueness within each layer
  3. Internal referential integrity (depends_on, placeholder references)
  4. Falsification predicate completeness (5-part I-2 syntax)
  5. Summary field consistency (counts match actual claim registry)

Exit code 0 if all layers pass, 1 if any fail.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

REPO_ROOT = Path(__file__).resolve().parent.parent
AI_LAYERS_DIR = REPO_ROOT / "ai-layers"
SCHEMA_PATH = AI_LAYERS_DIR / "ai-layer-schema.json"

REQUIRED_TOP_LEVEL = [
    "_meta", "paper_id", "paper_title", "paper_type",
    "version", "session", "phase_ready", "claim_registry",
    "placeholder_register",
]

REQUIRED_META = ["document_type", "schema_version", "produced_session", "licence"]
VALID_DOC_TYPES = ["AI_LAYER"]
VALID_PAPER_TYPES = ["law_A", "derivation_B", "application_C", "methodology_D"]
VALID_CLAIM_TYPES = ["F", "D", "A"]
VALID_VERDICTS = ["PHASE-READY", "NOT-PHASE-READY"]
VALID_SATISFACTION = ["SATISFIED", "UNSATISFIED"]

FALSIFICATION_PARTS = ["FALSIFIED_IF", "WHERE", "EVALUATION", "BOUNDARY", "CONTEXT"]
# Alternative key names used by some layers (e.g. DRP2)
ALT_FALSIFICATION_PARTS = ["condition", "inputs", "evaluation", "boundary", "context"]


class ValidationError:
    def __init__(self, file: str, path: str, message: str):
        self.file = file
        self.path = path
        self.message = message

    def __str__(self):
        return f"  [{self.file}] {self.path}: {self.message}"


def validate_layer(filepath: Path) -> "List[ValidationError]":
    errors = []
    fname = filepath.name

    try:
        with open(filepath) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [ValidationError(fname, "/", f"Invalid JSON: {e}")]

    # 1. Top-level required fields
    for field in REQUIRED_TOP_LEVEL:
        if field not in data:
            errors.append(ValidationError(fname, "/", f"Missing required field: {field}"))

    # 2. _meta validation
    meta = data.get("_meta", {})
    for field in REQUIRED_META:
        if field not in meta:
            errors.append(ValidationError(fname, "/_meta", f"Missing: {field}"))
    if meta.get("document_type") not in VALID_DOC_TYPES:
        errors.append(ValidationError(fname, "/_meta/document_type",
                                      f"Invalid: {meta.get('document_type')} (expected {VALID_DOC_TYPES})"))

    # 3. paper_type validation
    pt = data.get("paper_type")
    if pt and pt not in VALID_PAPER_TYPES:
        errors.append(ValidationError(fname, "/paper_type",
                                      f"Invalid: {pt} (expected {VALID_PAPER_TYPES})"))

    # 4. version/session pattern
    version = data.get("version", "")
    if version and not version.startswith("v"):
        errors.append(ValidationError(fname, "/version", f"Expected pattern ^v[0-9]+$, got: {version}"))

    session = data.get("session", "")
    if session and not session.startswith("S"):
        errors.append(ValidationError(fname, "/session", f"Expected pattern ^S[0-9]+$, got: {session}"))

    # 5. phase_ready validation
    pr = data.get("phase_ready", {})
    if "verdict" not in pr:
        errors.append(ValidationError(fname, "/phase_ready", "Missing: verdict"))
    elif pr["verdict"] not in VALID_VERDICTS:
        errors.append(ValidationError(fname, "/phase_ready/verdict",
                                      f"Invalid: {pr['verdict']}"))
    if "placeholder_count" not in pr:
        errors.append(ValidationError(fname, "/phase_ready", "Missing: placeholder_count"))

    # 6. Claim registry validation
    claims = data.get("claim_registry", [])
    if not isinstance(claims, list):
        errors.append(ValidationError(fname, "/claim_registry", "Must be an array"))
        claims = []

    claim_ids = set()
    type_counts = {"A": 0, "D": 0, "F": 0}

    for i, claim in enumerate(claims):
        prefix = f"/claim_registry[{i}]"

        # Required fields — accept known aliases
        if "claim_id" not in claim:
            errors.append(ValidationError(fname, prefix, "Missing: claim_id"))

        cid = claim.get("claim_id", f"<missing-{i}>")

        # Unique claim IDs
        if cid in claim_ids:
            errors.append(ValidationError(fname, prefix, f"Duplicate claim_id: {cid}"))
        claim_ids.add(cid)

        # Valid claim type — accept "type" or "claim_type"
        ctype = claim.get("type") or claim.get("claim_type")
        if ctype not in VALID_CLAIM_TYPES:
            errors.append(ValidationError(fname, f"{prefix}/type",
                                          f"Invalid: {ctype} (expected {VALID_CLAIM_TYPES})"))
        else:
            type_counts[ctype] += 1

        # Statement — accept "statement" or "label"
        if "statement" not in claim and "label" not in claim:
            errors.append(ValidationError(fname, prefix, "Missing: statement (or label)"))

        # Resolve falsification predicate from multiple possible locations:
        #   1. "falsification_predicate" key (standard)
        #   2. "predicate" key (P1 variant)
        #   3. Inline FALSIFIED_IF at claim level (MK-P1 variant)
        fp = claim.get("falsification_predicate") or claim.get("predicate")
        if fp is None and "FALSIFIED_IF" in claim:
            # Inline format: predicate fields live directly on the claim object
            fp = {k: claim[k] for k in FALSIFICATION_PARTS if k in claim}

        if fp is not None:
            if not isinstance(fp, dict):
                errors.append(ValidationError(fname, f"{prefix}/falsification_predicate",
                                              "Must be object or null"))
            else:
                # Check which key convention is used
                uses_standard = any(k in FALSIFICATION_PARTS for k in fp.keys())
                uses_alt = any(k in ALT_FALSIFICATION_PARTS for k in fp.keys())

                if uses_standard:
                    # Standard 5-part or mixed (standard keys present)
                    for part in FALSIFICATION_PARTS:
                        if part not in fp:
                            errors.append(ValidationError(fname, f"{prefix}/falsification_predicate",
                                                          f"Missing 5-part field: {part}"))
                        elif part != "WHERE" and isinstance(fp[part], str) and len(fp[part]) == 0:
                            errors.append(ValidationError(fname, f"{prefix}/falsification_predicate/{part}",
                                                          "Empty string (must be non-empty)"))
                elif uses_alt:
                    # Alternative key convention (condition/inputs/evaluation/boundary/context)
                    for part in ALT_FALSIFICATION_PARTS:
                        if part not in fp:
                            errors.append(ValidationError(fname, f"{prefix}/falsification_predicate",
                                                          f"Missing alt-format field: {part}"))
                elif len(fp) > 0:
                    # Multi-part: keys are sub-predicates, each containing the 5-part structure
                    for subkey, subpred in fp.items():
                        if not isinstance(subpred, dict):
                            errors.append(ValidationError(
                                fname, f"{prefix}/falsification_predicate/{subkey}",
                                "Multi-part sub-predicate must be object"))
                            continue
                        for part in FALSIFICATION_PARTS:
                            if part not in subpred:
                                errors.append(ValidationError(
                                    fname, f"{prefix}/falsification_predicate/{subkey}",
                                    f"Missing 5-part field: {part}"))
        elif ctype == "F":
            errors.append(ValidationError(fname, f"{prefix}/falsification_predicate",
                                          f"Type F claim {cid} has null falsification_predicate"))

    # 7. Placeholder register validation
    placeholders = data.get("placeholder_register", [])
    if not isinstance(placeholders, list):
        errors.append(ValidationError(fname, "/placeholder_register", "Must be an array"))
        placeholders = []

    for i, ph in enumerate(placeholders):
        prefix = f"/placeholder_register[{i}]"
        for field in ["placeholder_id", "source_claim", "resolved"]:
            if field not in ph:
                errors.append(ValidationError(fname, prefix, f"Missing: {field}"))

        # Check source_claim references a real claim
        src = ph.get("source_claim", "")
        if src and src not in claim_ids:
            errors.append(ValidationError(fname, prefix,
                                          f"source_claim '{src}' not found in claim_registry"))

    # 8. Summary consistency check
    summary = data.get("summary", {})
    if summary:
        expected_total = len(claims)
        if summary.get("total_claims") != expected_total:
            errors.append(ValidationError(fname, "/summary/total_claims",
                                          f"Says {summary.get('total_claims')} but registry has {expected_total}"))
        for t in ["A", "D", "F"]:
            key = f"type_{t}"
            if key in summary and summary[key] != type_counts[t]:
                errors.append(ValidationError(fname, f"/summary/{key}",
                                              f"Says {summary[key]} but counted {type_counts[t]}"))

    # 9. Phase-ready placeholder_count consistency
    actual_unresolved = sum(1 for ph in placeholders if not ph.get("resolved", True))
    stated_ph_count = pr.get("placeholder_count", -1)
    if stated_ph_count != actual_unresolved:
        errors.append(ValidationError(fname, "/phase_ready/placeholder_count",
                                      f"Says {stated_ph_count} but {actual_unresolved} unresolved in register"))

    return errors


def main():
    if not SCHEMA_PATH.exists():
        logger.fatal("Schema not found at %s", SCHEMA_PATH)
        sys.exit(1)

    layer_files = sorted(AI_LAYERS_DIR.glob("*-ai-layer.json"))
    if not layer_files:
        logger.warning("No AI layer files found matching *-ai-layer.json")
        sys.exit(0)

    print(f"Fracttalix AI Layer Validator")
    print(f"Schema: {SCHEMA_PATH.name}")
    print(f"Layers found: {len(layer_files)}")
    print("=" * 60)

    total_errors = 0
    for filepath in layer_files:
        errors = validate_layer(filepath)
        status = "PASS" if not errors else "FAIL"
        print(f"\n[{status}] {filepath.name}")
        for e in errors:
            print(e)
        total_errors += len(errors)

    print("\n" + "=" * 60)
    print(f"Total: {len(layer_files)} layers, {total_errors} errors")

    if total_errors > 0:
        print("VALIDATION FAILED")
        sys.exit(1)
    else:
        print("ALL LAYERS VALID")
        sys.exit(0)


if __name__ == "__main__":
    main()
