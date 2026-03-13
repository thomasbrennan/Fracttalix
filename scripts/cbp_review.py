#!/usr/bin/env python3
"""CBP (Canonical Build Process) integration for team orchestration.

Implements the five-phase Meta Kaizen review cycle as message-bus protocol:

  Phase 1: FIRST_BUILD_PLAN  — Coordinator decomposes work into tasks with
                                falsifiable acceptance criteria (claims, structure)
  Phase 2: FIRST_META_KAIZEN — Executor self-reviews using KVS-based
                                improvement scoring BEFORE hostile review.
                                Identifies weaknesses proactively and fixes them.
  Phase 3: HOSTILE_REVIEW    — Verifier raises adversarial objections against
                                each deliverable (8-10 per deliverable, rated
                                HIGH/MEDIUM/LOW severity)
  Phase 4: SECOND_META_KAIZEN — Executor responds to each objection with a
                                categorized resolution: strengthened, resolved_stronger,
                                discipline_enforced, scope_refined, or fixed
  Phase 5: FINAL_BUILD_PLAN  — Coordinator evaluates. Accepts only after all
                                HIGH-severity objections are resolved. Documents
                                all changes and produces final deliverable.

This module adds CBP-specific message types to the message bus and provides
a review state machine that tracks deliverables through the cycle.

Session S57 — 2026-03-13
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import sibling modules
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
import message_bus  # noqa: E402

_REPO_ROOT = _SCRIPTS.parent
REVIEW_DIR = _REPO_ROOT / ".checkpoint" / "reviews"

# ---------------------------------------------------------------------------
# CBP message types (extensions to the base message bus)
# ---------------------------------------------------------------------------

# Phase 1: First Build Plan — Coordinator → Executor
MSG_DELIVERABLE_SPEC = "cbp_deliverable_spec"

# Phase 2: First Meta Kaizen — Executor → Coordinator (self-review)
MSG_SELF_REVIEW = "cbp_self_review"

# Phase 3: Hostile Review — Coordinator → Verifier (request), Verifier → Coordinator (objections)
MSG_REVIEW_REQUEST = "cbp_review_request"
MSG_HOSTILE_REVIEW = "cbp_hostile_review"

# Phase 4: Second Meta Kaizen — Coordinator → Executor (forward), Executor → Coordinator (responses)
MSG_OBJECTIONS_FORWARD = "cbp_objections_forward"
MSG_REVIEW_RESPONSE = "cbp_review_response"

# Phase 5: Final Build Plan — Coordinator → Executor (accept/reject)
MSG_FINAL_ACCEPT = "cbp_final_accept"
MSG_FINAL_REJECT = "cbp_final_reject"

# Objection severity levels
SEVERITY_HIGH = "HIGH"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_LOW = "LOW"

# Response categories (from Meta Kaizen Phase 3)
RESPONSE_STRENGTHENED = "strengthened"
RESPONSE_RESOLVED_STRONGER = "resolved_stronger"
RESPONSE_DISCIPLINE_ENFORCED = "discipline_enforced"
RESPONSE_SCOPE_REFINED = "scope_refined"
RESPONSE_FIXED = "fixed"

VALID_RESPONSE_CATEGORIES = frozenset({
    RESPONSE_STRENGTHENED,
    RESPONSE_RESOLVED_STRONGER,
    RESPONSE_DISCIPLINE_ENFORCED,
    RESPONSE_SCOPE_REFINED,
    RESPONSE_FIXED,
})

# Review states (five-phase cycle)
STATE_SPEC = "spec"                         # Phase 1: First Build Plan created
STATE_SELF_REVIEWED = "self_reviewed"       # Phase 2: First Meta Kaizen complete
STATE_UNDER_REVIEW = "under_review"         # Phase 3: Hostile Review in progress
STATE_RESPONDING = "responding"             # Phase 4: Second Meta Kaizen in progress
STATE_ACCEPTED = "accepted"                 # Phase 5: Final Build Plan — accepted
STATE_REJECTED = "rejected"                 # Phase 5: Final Build Plan — rejected


# ---------------------------------------------------------------------------
# Review record management
# ---------------------------------------------------------------------------

def _ensure_dir() -> None:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)


def _review_path(task_id: str) -> Path:
    return REVIEW_DIR / f"{task_id}.json"


def create_review(
    task_id: str,
    deliverable_desc: str,
    acceptance_criteria: List[str],
    executor: str,
) -> Dict[str, Any]:
    """Phase 1: Create a review record for a deliverable."""
    _ensure_dir()
    review = {
        "task_id": task_id,
        "deliverable": deliverable_desc,
        "acceptance_criteria": acceptance_criteria,
        "executor": executor,
        "state": STATE_SPEC,
        "objections": [],
        "responses": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "phase_history": [
            {"phase": 1, "state": STATE_SPEC,
             "at": datetime.now(timezone.utc).isoformat()}
        ],
    }
    _review_path(task_id).write_text(json.dumps(review, indent=2) + "\n")
    return review


def load_review(task_id: str) -> Dict[str, Any]:
    """Load a review record."""
    path = _review_path(task_id)
    if not path.exists():
        raise FileNotFoundError(f"No review record for task {task_id}")
    return json.loads(path.read_text())


def _save_review(review: Dict[str, Any]) -> None:
    review["updated_at"] = datetime.now(timezone.utc).isoformat()
    _review_path(review["task_id"]).write_text(
        json.dumps(review, indent=2) + "\n"
    )


# ---------------------------------------------------------------------------
# Phase 2: First Meta Kaizen (self-review before hostile review)
# ---------------------------------------------------------------------------

def submit_self_review(
    task_id: str,
    improvements: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Phase 2: Executor submits self-review (First Meta Kaizen).

    Before facing hostile review, the executor applies KVS-based
    improvement scoring to their own work and proactively fixes issues.

    Each improvement: {
        "id": "I-1",
        "area": "clarity|correctness|completeness|robustness|efficiency",
        "finding": "what was found",
        "action_taken": "what was done to improve it",
        "kvs_delta": "+1"  # estimated improvement score
    }
    """
    review = load_review(task_id)
    if review["state"] != STATE_SPEC:
        raise ValueError(
            f"Self-review requires state '{STATE_SPEC}', got '{review['state']}'"
        )

    review["self_review"] = {
        "improvements": improvements,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "improvement_count": len(improvements),
    }
    review["state"] = STATE_SELF_REVIEWED
    review["phase_history"].append({
        "phase": 2, "state": STATE_SELF_REVIEWED,
        "at": datetime.now(timezone.utc).isoformat(),
        "improvement_count": len(improvements),
    })
    _save_review(review)

    # Notify coordinator that self-review is complete → ready for hostile review
    message_bus.send_message(
        review["executor"], "coordinator", MSG_SELF_REVIEW,
        {"task_id": task_id, "improvement_count": len(improvements)}
    )
    return review


# ---------------------------------------------------------------------------
# Phase 3: Hostile Review
# ---------------------------------------------------------------------------

def submit_hostile_review(
    task_id: str,
    objections: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Phase 2: Verifier submits hostile review objections.

    Each objection: {"id": "O-1", "text": "...", "severity": "HIGH|MEDIUM|LOW"}
    """
    review = load_review(task_id)
    if review["state"] not in (STATE_SELF_REVIEWED, STATE_UNDER_REVIEW):
        raise ValueError(
            f"Cannot submit hostile review in state '{review['state']}'. "
            f"Self-review (Phase 2: First Meta Kaizen) must complete first."
        )

    for obj in objections:
        if "id" not in obj or "text" not in obj or "severity" not in obj:
            raise ValueError(
                f"Objection must have id, text, severity. Got: {obj}"
            )
        if obj["severity"] not in (SEVERITY_HIGH, SEVERITY_MEDIUM, SEVERITY_LOW):
            raise ValueError(f"Invalid severity: {obj['severity']}")

    review["objections"] = objections
    review["state"] = STATE_UNDER_REVIEW
    review["phase_history"].append({
        "phase": 3, "state": STATE_UNDER_REVIEW,
        "at": datetime.now(timezone.utc).isoformat(),
        "objection_count": len(objections),
        "high_count": sum(1 for o in objections if o["severity"] == SEVERITY_HIGH),
    })
    _save_review(review)

    # Send via message bus: verifier → coordinator
    message_bus.send_message(
        "verifier", "coordinator", MSG_HOSTILE_REVIEW,
        {"task_id": task_id, "objection_count": len(objections)}
    )
    return review


# ---------------------------------------------------------------------------
# Phase 4: Second Meta Kaizen (respond to hostile review objections)
# ---------------------------------------------------------------------------

def submit_response(
    task_id: str,
    objection_id: str,
    category: str,
    explanation: str,
    changes_made: Optional[str] = None,
) -> Dict[str, Any]:
    """Phase 3: Executor responds to a specific objection.

    category must be one of: strengthened, resolved_stronger,
    discipline_enforced, scope_refined, fixed
    """
    if category not in VALID_RESPONSE_CATEGORIES:
        raise ValueError(
            f"Invalid response category '{category}'. "
            f"Must be one of: {sorted(VALID_RESPONSE_CATEGORIES)}"
        )

    review = load_review(task_id)
    if review["state"] not in (STATE_UNDER_REVIEW, STATE_RESPONDING):
        raise ValueError(
            f"Cannot submit response in state '{review['state']}'"
        )

    # Verify objection exists
    obj_ids = {o["id"] for o in review["objections"]}
    if objection_id not in obj_ids:
        raise ValueError(f"Unknown objection: {objection_id}")

    response = {
        "objection_id": objection_id,
        "category": category,
        "explanation": explanation,
        "changes_made": changes_made,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    review["responses"].append(response)
    review["state"] = STATE_RESPONDING
    _save_review(review)
    return review


# ---------------------------------------------------------------------------
# Phase 5: Final Build Plan — Accept / Reject
# ---------------------------------------------------------------------------

def evaluate_review(task_id: str) -> Dict[str, Any]:
    """Phase 4: Evaluate whether all objections are resolved.

    Returns the review with state set to 'accepted' or 'rejected'.
    Rejection happens if any HIGH-severity objection lacks a response.
    """
    review = load_review(task_id)

    responded_ids = {r["objection_id"] for r in review["responses"]}
    high_objections = [
        o for o in review["objections"] if o["severity"] == SEVERITY_HIGH
    ]
    unresolved_high = [
        o for o in high_objections if o["id"] not in responded_ids
    ]

    all_objections = review["objections"]
    unresolved_all = [
        o for o in all_objections if o["id"] not in responded_ids
    ]

    if unresolved_high:
        review["state"] = STATE_REJECTED
        review["phase_history"].append({
            "phase": 5, "state": STATE_REJECTED,
            "at": datetime.now(timezone.utc).isoformat(),
            "reason": f"{len(unresolved_high)} unresolved HIGH objections",
            "unresolved": [o["id"] for o in unresolved_high],
        })
    else:
        review["state"] = STATE_ACCEPTED
        review["phase_history"].append({
            "phase": 5, "state": STATE_ACCEPTED,
            "at": datetime.now(timezone.utc).isoformat(),
            "resolved_count": len(review["responses"]),
            "unresolved_low_medium": len(unresolved_all),
        })

    _save_review(review)

    # Notify via message bus
    msg_type = MSG_FINAL_ACCEPT if review["state"] == STATE_ACCEPTED else MSG_FINAL_REJECT
    message_bus.send_message(
        "coordinator", review["executor"], msg_type,
        {
            "task_id": task_id,
            "state": review["state"],
            "resolved": len(review["responses"]),
            "unresolved_high": len(unresolved_high),
        }
    )
    return review


def get_review_summary(task_id: str) -> str:
    """Human-readable summary of a review's current state."""
    review = load_review(task_id)
    lines = [
        f"Task: {review['task_id']}",
        f"State: {review['state']}",
        f"Deliverable: {review['deliverable']}",
        f"Executor: {review['executor']}",
        f"Objections: {len(review['objections'])}",
    ]
    if review["objections"]:
        high = sum(1 for o in review["objections"] if o["severity"] == SEVERITY_HIGH)
        med = sum(1 for o in review["objections"] if o["severity"] == SEVERITY_MEDIUM)
        low = sum(1 for o in review["objections"] if o["severity"] == SEVERITY_LOW)
        lines.append(f"  HIGH: {high}, MEDIUM: {med}, LOW: {low}")

    responded_ids = {r["objection_id"] for r in review["responses"]}
    lines.append(f"Responses: {len(review['responses'])}/{len(review['objections'])}")

    for obj in review["objections"]:
        status = "RESOLVED" if obj["id"] in responded_ids else "OPEN"
        lines.append(f"  [{obj['severity']}] {obj['id']}: {obj['text'][:60]}... [{status}]")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CBP (Canonical Build Process) review pipeline"
    )
    sub = parser.add_subparsers(dest="command")

    # -- create --
    p_create = sub.add_parser("create", help="Phase 1: Create review for a task")
    p_create.add_argument("--task", required=True, help="Task ID (e.g. T-001)")
    p_create.add_argument("--desc", required=True, help="Deliverable description")
    p_create.add_argument("--criteria", required=True,
                          help="Comma-separated acceptance criteria")
    p_create.add_argument("--executor", required=True, help="Assigned executor role")

    # -- self-review --
    p_self = sub.add_parser("self-review",
                            help="Phase 2: Submit first Meta Kaizen self-review")
    p_self.add_argument("--task", required=True)
    p_self.add_argument("--improvements", required=True,
                        help="JSON array of improvements")

    # -- review --
    p_review = sub.add_parser("review", help="Phase 3: Submit hostile review")
    p_review.add_argument("--task", required=True)
    p_review.add_argument("--objections", required=True,
                          help="JSON array of objections")

    # -- respond --
    p_respond = sub.add_parser("respond", help="Phase 3: Respond to an objection")
    p_respond.add_argument("--task", required=True)
    p_respond.add_argument("--objection", required=True, help="Objection ID")
    p_respond.add_argument("--category", required=True,
                           choices=sorted(VALID_RESPONSE_CATEGORIES))
    p_respond.add_argument("--explanation", required=True)
    p_respond.add_argument("--changes", default=None)

    # -- evaluate --
    p_eval = sub.add_parser("evaluate", help="Phase 4: Evaluate review outcome")
    p_eval.add_argument("--task", required=True)

    # -- status --
    p_status = sub.add_parser("status", help="Show review status")
    p_status.add_argument("--task", required=True)

    # -- list --
    sub.add_parser("list", help="List all reviews")

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "create":
        criteria = [c.strip() for c in args.criteria.split(",")]
        review = create_review(args.task, args.desc, criteria, args.executor)
        print(json.dumps(review, indent=2))

    elif args.command == "self-review":
        improvements = json.loads(args.improvements)
        review = submit_self_review(args.task, improvements)
        print(f"Self-review recorded: {len(improvements)} improvements")
        print(json.dumps(review["self_review"], indent=2))

    elif args.command == "review":
        objections = json.loads(args.objections)
        review = submit_hostile_review(args.task, objections)
        print(json.dumps(review, indent=2))

    elif args.command == "respond":
        review = submit_response(
            args.task, args.objection, args.category,
            args.explanation, args.changes
        )
        print(f"Response recorded for {args.objection}")

    elif args.command == "evaluate":
        review = evaluate_review(args.task)
        print(get_review_summary(args.task))

    elif args.command == "status":
        print(get_review_summary(args.task))

    elif args.command == "list":
        _ensure_dir()
        reviews = sorted(REVIEW_DIR.glob("*.json"))
        if not reviews:
            print("No reviews found.")
        for path in reviews:
            review = json.loads(path.read_text())
            print(f"  {review['task_id']:8s} [{review['state']:14s}] {review['deliverable'][:50]}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
