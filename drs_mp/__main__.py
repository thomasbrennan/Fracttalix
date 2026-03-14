"""CLI for DRS-MP message validation and inspection.

Usage:
    python -m drs_mp validate relay/queue/MSG-*.json
    python -m drs_mp inspect relay/queue/MSG-20260314-114429-6f40.json
    python -m drs_mp completeness relay/queue/MSG-response.json
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

from drs_mp import Message


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate DRS-MP messages against the protocol schema."""
    files = []
    for pattern in args.files:
        files.extend(glob.glob(pattern))
    if not files:
        print("No files found.", file=sys.stderr)
        return 1

    total_errors = 0
    for path in sorted(files):
        try:
            msg = Message.load(path)
            errors = msg.validate()
            if errors:
                print(f"FAIL  {path}")
                for e in errors:
                    print(f"      {e}")
                total_errors += len(errors)
            elif not args.quiet:
                print(f"OK    {path} ({msg.type}, {len(msg.claims)} claims, {len(msg.verdicts)} verdicts)")
        except Exception as e:
            print(f"ERROR {path}: {e}", file=sys.stderr)
            total_errors += 1

    if total_errors:
        print(f"\n{total_errors} error(s) in {len(files)} file(s).")
        return 1
    print(f"\nAll {len(files)} message(s) valid.")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect a DRS-MP message and display its structured content."""
    msg = Message.load(args.file)
    print(f"Message:  {msg.msg_id}")
    print(f"From:     {msg.from_agent}")
    print(f"To:       {msg.to_agent}")
    print(f"Type:     {msg.type}")
    print(f"Priority: {msg.priority}")
    print(f"Status:   {msg.status}")
    print(f"Subject:  {msg.subject}")
    print(f"Protocol: {msg.protocol_version}")

    if msg.claims:
        print(f"\nClaims ({len(msg.claims)}):")
        for c in msg.claims:
            tag = f"[{c.type}]"
            pred = " (has predicate)" if c.falsification_predicate else ""
            print(f"  {c.claim_id} {tag} {c.label or c.statement[:60]}{pred}")

    if msg.verdicts:
        print(f"\nVerdicts ({len(msg.verdicts)}):")
        for v in msg.verdicts:
            print(f"  {v.claim_id}: {v.verdict} (confidence: {v.confidence})")

    if msg.objections:
        print(f"\nObjections ({len(msg.objections)}):")
        for o in msg.objections:
            print(f"  {o.objection_id} -> {o.targets_claim} [{o.objection_type}] ({o.severity})")
            print(f"    {o.statement[:100]}")

    errors = msg.validate()
    if errors:
        print(f"\nValidation errors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
    else:
        print("\nValidation: PASS")

    return 0


def cmd_completeness(args: argparse.Namespace) -> int:
    """Check review completeness for a response message."""
    msg = Message.load(args.file)
    result = msg.check_review_completeness()
    print(f"Message:  {msg.msg_id}")
    print(f"Complete: {result['complete']}")
    print(f"Claims:   {result['total_claims']}")
    print(f"Verdicts: {result['total_verdicts']}")
    if result["reviewed"]:
        print(f"Reviewed: {', '.join(result['reviewed'])}")
    if result["missing"]:
        print(f"Missing:  {', '.join(result['missing'])}")
    return 0 if result["complete"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="drs-mp",
        description="DRS-MP — Dual Reader Standard Message Protocol tools",
    )
    sub = parser.add_subparsers(dest="command")

    p_val = sub.add_parser("validate", help="Validate DRS-MP message files")
    p_val.add_argument("files", nargs="+", help="Message JSON files (glob patterns supported)")
    p_val.add_argument("-q", "--quiet", action="store_true", help="Only show errors")

    p_insp = sub.add_parser("inspect", help="Inspect a DRS-MP message")
    p_insp.add_argument("file", help="Message JSON file")

    p_comp = sub.add_parser("completeness", help="Check review completeness")
    p_comp.add_argument("file", help="Response message JSON file")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    commands = {"validate": cmd_validate, "inspect": cmd_inspect, "completeness": cmd_completeness}
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
