"""Core DRS-MP message types — zero dependencies, stdlib only.

This module implements the Dual Reader Standard Message Protocol (DRS-MP v2)
for structured inter-AI communication. Every message carries two channels:
  - Channel 1 (prose body): Human-readable audit trail
  - Channel 2 (structured fields): Machine-authoritative typed claims,
    objections, and verdicts with falsification predicates

The protocol is transport-independent — it works over git relay, HTTP,
Google A2A, Anthropic MCP, or any other messaging layer.

Reference: https://github.com/thomasbrennan/Fracttalix/blob/main/relay/protocol-v2.json
"""
from __future__ import annotations

import json
import random
import string
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _generate_msg_id() -> str:
    now = datetime.now(timezone.utc)
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"MSG-{now.strftime('%Y%m%d-%H%M%S')}-{suffix}"


# --- Falsification Predicate ---

@dataclass
class FalsificationPredicate:
    """Five-part deterministic falsification predicate (Falsification Kernel).

    Every Type F claim must carry all five parts. The predicate evaluates to
    FALSIFIED or NOT FALSIFIED without natural language interpretation.

    Parts:
        FALSIFIED_IF: The specific observable condition that would falsify the claim.
        WHERE: All variables defined with types, units, and sources.
        EVALUATION: The exact procedure for testing (third-party executable).
        BOUNDARY: Numerical thresholds with justification.
        CONTEXT: Why the threshold is set where it is.
    """
    FALSIFIED_IF: str
    WHERE: str
    EVALUATION: str
    BOUNDARY: str
    CONTEXT: str

    def validate(self) -> list[str]:
        errors = []
        for part in ("FALSIFIED_IF", "WHERE", "EVALUATION", "BOUNDARY", "CONTEXT"):
            val = getattr(self, part)
            if not val or not val.strip():
                errors.append(f"Predicate {part} is empty")
        return errors

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> FalsificationPredicate:
        return cls(**{k: d[k] for k in ("FALSIFIED_IF", "WHERE", "EVALUATION", "BOUNDARY", "CONTEXT")})


# --- Claim ---

CLAIM_TYPES = {"A", "D", "F"}

@dataclass
class Claim:
    """A typed scientific claim with optional falsification predicate.

    Types:
        A — Axiom/assumption (no predicate required)
        D — Definition (no predicate required)
        F — Falsifiable (predicate REQUIRED — all 5 parts)
    """
    claim_id: str
    type: str  # "A", "D", or "F"
    statement: str
    label: str = ""
    source_paper: str = ""
    source_section: str = ""
    falsification_predicate: FalsificationPredicate | None = None

    @classmethod
    def falsifiable(
        cls,
        claim_id: str,
        statement: str,
        falsified_if: str,
        where: str,
        evaluation: str,
        boundary: str,
        context: str,
        **kwargs: Any,
    ) -> Claim:
        """Create a Type F claim with all 5 predicate parts."""
        return cls(
            claim_id=claim_id,
            type="F",
            statement=statement,
            falsification_predicate=FalsificationPredicate(
                FALSIFIED_IF=falsified_if,
                WHERE=where,
                EVALUATION=evaluation,
                BOUNDARY=boundary,
                CONTEXT=context,
            ),
            **kwargs,
        )

    @classmethod
    def axiom(cls, claim_id: str, statement: str, **kwargs: Any) -> Claim:
        return cls(claim_id=claim_id, type="A", statement=statement, **kwargs)

    @classmethod
    def definition(cls, claim_id: str, statement: str, **kwargs: Any) -> Claim:
        return cls(claim_id=claim_id, type="D", statement=statement, **kwargs)

    def validate(self) -> list[str]:
        errors = []
        if not self.claim_id:
            errors.append("Claim missing claim_id")
        if self.type not in CLAIM_TYPES:
            errors.append(f"Invalid claim type: {self.type} (must be A, D, or F)")
        if not self.statement:
            errors.append(f"Claim {self.claim_id}: missing statement")
        if self.type == "F":
            if not self.falsification_predicate:
                errors.append(f"Claim {self.claim_id}: Type F requires falsification_predicate")
            else:
                errors.extend(
                    f"Claim {self.claim_id}: {e}"
                    for e in self.falsification_predicate.validate()
                )
        return errors

    def to_dict(self) -> dict:
        d = {"claim_id": self.claim_id, "type": self.type, "statement": self.statement}
        if self.label:
            d["label"] = self.label
        if self.source_paper:
            d["source_paper"] = self.source_paper
        if self.source_section:
            d["source_section"] = self.source_section
        if self.falsification_predicate:
            d["falsification_predicate"] = self.falsification_predicate.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Claim:
        fp = None
        if "falsification_predicate" in d and d["falsification_predicate"]:
            fp = FalsificationPredicate.from_dict(d["falsification_predicate"])
        return cls(
            claim_id=d["claim_id"],
            type=d["type"],
            statement=d["statement"],
            label=d.get("label", ""),
            source_paper=d.get("source_paper", ""),
            source_section=d.get("source_section", ""),
            falsification_predicate=fp,
        )


# --- Verdict ---

VERDICT_VALUES = {"confirmed", "disputed", "inconclusive", "needs-revision"}

@dataclass
class Verdict:
    """Machine-parseable review verdict for a specific claim."""
    claim_id: str
    verdict: str  # confirmed | disputed | inconclusive | needs-revision
    confidence: float  # 0.0 to 1.0
    reasoning: str = ""
    predicate_assessment: dict = field(default_factory=dict)
    sources_checked: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        errors = []
        if not self.claim_id:
            errors.append("Verdict missing claim_id")
        if self.verdict not in VERDICT_VALUES:
            errors.append(f"Invalid verdict: {self.verdict}")
        if not (0.0 <= self.confidence <= 1.0):
            errors.append(f"Confidence {self.confidence} out of range [0, 1]")
        return errors

    def to_dict(self) -> dict:
        d = {"claim_id": self.claim_id, "verdict": self.verdict, "confidence": self.confidence}
        if self.reasoning:
            d["reasoning"] = self.reasoning
        if self.predicate_assessment:
            d["predicate_assessment"] = self.predicate_assessment
        if self.sources_checked:
            d["sources_checked"] = self.sources_checked
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Verdict:
        return cls(
            claim_id=d["claim_id"],
            verdict=d["verdict"],
            confidence=d["confidence"],
            reasoning=d.get("reasoning", ""),
            predicate_assessment=d.get("predicate_assessment", {}),
            sources_checked=d.get("sources_checked", []),
        )


# --- Objection ---

OBJECTION_TYPES = {
    "logical-gap", "counterexample", "unstated-assumption", "vacuity",
    "circularity", "scope-overreach", "empirical-gap", "prior-art-overlap",
    "definition-weakness",
}
SEVERITY_LEVELS = {"critical", "major", "minor"}

@dataclass
class Objection:
    """Typed hostile review objection targeting a specific claim."""
    objection_id: str
    targets_claim: str
    objection_type: str
    statement: str
    severity: str  # critical | major | minor
    proposed_test: str = ""

    def validate(self) -> list[str]:
        errors = []
        if not self.objection_id:
            errors.append("Objection missing objection_id")
        if not self.targets_claim:
            errors.append(f"Objection {self.objection_id}: missing targets_claim")
        if self.objection_type not in OBJECTION_TYPES:
            errors.append(f"Invalid objection_type: {self.objection_type}")
        if self.severity not in SEVERITY_LEVELS:
            errors.append(f"Invalid severity: {self.severity}")
        if not self.statement:
            errors.append(f"Objection {self.objection_id}: missing statement")
        return errors

    def to_dict(self) -> dict:
        d = {
            "objection_id": self.objection_id,
            "targets_claim": self.targets_claim,
            "objection_type": self.objection_type,
            "statement": self.statement,
            "severity": self.severity,
        }
        if self.proposed_test:
            d["proposed_test"] = self.proposed_test
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Objection:
        return cls(
            objection_id=d["objection_id"],
            targets_claim=d["targets_claim"],
            objection_type=d["objection_type"],
            statement=d["statement"],
            severity=d["severity"],
            proposed_test=d.get("proposed_test", ""),
        )


# --- Message ---

MESSAGE_TYPES = {
    "claim-review", "claim-review-response",
    "hostile-review", "hostile-review-response",
    "cross-reference", "cross-reference-response",
    "standing-order", "status-query", "status-response", "general",
}
PRIORITY_LEVELS = {"low", "normal", "high", "critical"}
STATUS_VALUES = {"pending", "acknowledged", "in-progress", "resolved", "expired"}

@dataclass
class Message:
    """DRS-MP v2 message — dual-channel inter-AI communication.

    Channel 1 (body): Human-readable prose for audit trails.
    Channel 2 (claims/verdicts/objections): Machine-authoritative structured content.
    """
    msg_id: str
    timestamp: str
    from_agent: str
    to_agent: str
    type: str
    subject: str
    body: str
    priority: str = "normal"
    protocol_version: str = "2.0.0"
    status: str = "pending"
    claims: list[Claim] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)
    objections: list[Objection] = field(default_factory=list)
    references: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        from_agent: str,
        to_agent: str,
        type: str,
        subject: str,
        body: str,
        **kwargs: Any,
    ) -> Message:
        return cls(
            msg_id=_generate_msg_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            from_agent=from_agent,
            to_agent=to_agent,
            type=type,
            subject=subject,
            body=body,
            **kwargs,
        )

    @classmethod
    def review_request(
        cls,
        from_agent: str,
        to_agent: str,
        subject: str,
        claims: list[Claim],
        body: str = "",
        **kwargs: Any,
    ) -> Message:
        """Create a hostile review request with structured claims."""
        if not body:
            body = f"Hostile review request: {subject}. {len(claims)} claim(s) for adversarial review. Your objective is to FALSIFY, not confirm."
        return cls.create(
            from_agent=from_agent,
            to_agent=to_agent,
            type="hostile-review",
            subject=subject,
            body=body,
            claims=claims,
            priority="high",
            **kwargs,
        )

    @classmethod
    def review_response(
        cls,
        original: Message,
        from_agent: str,
        verdicts: list[Verdict],
        objections: list[Objection] | None = None,
        body: str = "",
        **kwargs: Any,
    ) -> Message:
        """Create a structured review response to a review request."""
        if not body:
            body = f"Review response: {len(verdicts)} verdict(s), {len(objections or [])} objection(s)."
        return cls.create(
            from_agent=from_agent,
            to_agent=original.from_agent,
            type=f"{original.type}-response",
            subject=f"Re: {original.subject}",
            body=body,
            verdicts=verdicts,
            objections=objections or [],
            references={"parent_message": original.msg_id},
            priority=original.priority,
            **kwargs,
        )

    def validate(self) -> list[str]:
        """Validate the message against DRS-MP v2 schema. Returns list of errors."""
        errors = []
        if not self.msg_id:
            errors.append("Missing msg_id")
        if not self.from_agent:
            errors.append("Missing from_agent")
        if not self.to_agent:
            errors.append("Missing to_agent")
        if self.type not in MESSAGE_TYPES:
            errors.append(f"Invalid message type: {self.type}")
        if self.priority not in PRIORITY_LEVELS:
            errors.append(f"Invalid priority: {self.priority}")
        if self.status not in STATUS_VALUES:
            errors.append(f"Invalid status: {self.status}")
        if not self.subject:
            errors.append("Missing subject")
        if not self.body:
            errors.append("Missing body (Channel 1)")

        for claim in self.claims:
            errors.extend(claim.validate())
        for verdict in self.verdicts:
            errors.extend(verdict.validate())
        for objection in self.objections:
            errors.extend(objection.validate())

        return errors

    def check_review_completeness(self) -> dict:
        """Check if all claims in the request have corresponding verdicts.

        Returns dict with 'complete' bool, 'reviewed' and 'missing' claim IDs.
        This is the automated completeness verification described in F-MK8.2.
        """
        claim_ids = {c.claim_id for c in self.claims}
        verdict_ids = {v.claim_id for v in self.verdicts}

        # Check against references if no claims in this message
        if not claim_ids and self.references.get("claim_ids"):
            claim_ids = set(self.references["claim_ids"])

        reviewed = claim_ids & verdict_ids
        missing = claim_ids - verdict_ids

        return {
            "complete": len(missing) == 0 and len(claim_ids) > 0,
            "reviewed": sorted(reviewed),
            "missing": sorted(missing),
            "total_claims": len(claim_ids),
            "total_verdicts": len(verdict_ids),
        }

    def to_dict(self) -> dict:
        d = {
            "msg_id": self.msg_id,
            "timestamp": self.timestamp,
            "from": self.from_agent,
            "to": self.to_agent,
            "type": self.type,
            "subject": self.subject,
            "body": self.body,
            "priority": self.priority,
            "protocol_version": self.protocol_version,
            "status": self.status,
        }
        if self.claims:
            d["claims"] = [c.to_dict() for c in self.claims]
        if self.verdicts:
            d["verdicts"] = [v.to_dict() for v in self.verdicts]
        if self.objections:
            d["objections"] = [o.to_dict() for o in self.objections]
        if self.references:
            d["references"] = self.references
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, directory: str | Path) -> Path:
        """Save message to a JSON file in the given directory."""
        path = Path(directory) / f"{self.msg_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path

    @classmethod
    def from_dict(cls, d: dict) -> Message:
        claims = [Claim.from_dict(c) for c in d.get("claims", [])]
        verdicts = [Verdict.from_dict(v) for v in d.get("verdicts", [])]
        objections = [Objection.from_dict(o) for o in d.get("objections", [])]
        return cls(
            msg_id=d["msg_id"],
            timestamp=d["timestamp"],
            from_agent=d["from"],
            to_agent=d["to"],
            type=d["type"],
            subject=d["subject"],
            body=d["body"],
            priority=d.get("priority", "normal"),
            protocol_version=d.get("protocol_version", "2.0.0"),
            status=d.get("status", "pending"),
            claims=claims,
            verdicts=verdicts,
            objections=objections,
            references=d.get("references", {}),
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> Message:
        """Load a message from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_json(cls, text: str) -> Message:
        return cls.from_dict(json.loads(text))
