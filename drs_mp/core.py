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
VERIFICATION_TIERS = {
    "axiom", "definition", "software_tested", "analytic",
    "empirical_pending", "formal_proof",
}

@dataclass
class Claim:
    """A typed scientific claim with optional falsification predicate and GVP fields.

    Types:
        A — Axiom/assumption (no predicate required)
        D — Definition (no predicate required)
        F — Falsifiable (predicate REQUIRED — all 5 parts)

    GVP (Grounded Verification Protocol) fields for software claims:
        tier — verification tier (axiom, definition, software_tested, analytic, empirical_pending, formal_proof)
        test_bindings — list of fully qualified pytest node IDs that verify this claim
        verified_against — git commit SHA at which test_bindings last passed
    """
    claim_id: str
    type: str  # "A", "D", or "F"
    statement: str
    label: str = ""
    source_paper: str = ""
    source_section: str = ""
    falsification_predicate: FalsificationPredicate | None = None
    # GVP fields
    tier: str = ""  # verification tier
    test_bindings: list[str] = field(default_factory=list)
    verified_against: str = ""  # git commit SHA

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
        # GVP validation
        if self.tier and self.tier not in VERIFICATION_TIERS:
            errors.append(f"Claim {self.claim_id}: invalid GVP tier '{self.tier}'")
        if self.tier == "software_tested":
            if not self.test_bindings:
                errors.append(f"Claim {self.claim_id}: tier 'software_tested' requires non-empty test_bindings")
            if not self.verified_against:
                errors.append(f"Claim {self.claim_id}: tier 'software_tested' requires verified_against SHA")
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
        if self.tier:
            d["tier"] = self.tier
        if self.test_bindings:
            d["test_bindings"] = self.test_bindings
        if self.verified_against:
            d["verified_against"] = self.verified_against
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
            tier=d.get("tier", ""),
            test_bindings=d.get("test_bindings", []),
            verified_against=d.get("verified_against", ""),
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


# --- Protocol Spec (embedded in every message) ---

PROTOCOL_SPEC = {
    "protocol": "DRS-MP",
    "version": "2.0.0",
    "name": "Dual Reader Standard Message Protocol",
    "purpose": "Structured inter-AI communication with epistemologically typed claims. Every message carries two channels: Channel 1 (prose body) for human audit trails, Channel 2 (structured fields) for machine-authoritative content.",
    "how_to_respond": {
        "step_1": "Parse this JSON message. The 'claims' array contains typed scientific claims with falsification predicates.",
        "step_2": "For each claim, evaluate independently. Your objective is to FALSIFY, not confirm.",
        "step_3": "Respond with a JSON object containing a 'verdicts' array — one verdict per claim.",
        "step_4": "Each verdict must include: claim_id, verdict (confirmed|disputed|inconclusive|needs-revision), confidence (0.0-1.0), and reasoning.",
        "step_5": "If you find defects, include an 'objections' array with typed objections (objection_id, targets_claim, objection_type, statement, severity).",
        "step_6": "Output ONLY valid JSON. No markdown fences, no preamble.",
    },
    "claim_types": {
        "A": "Axiom/assumption — no falsification predicate required",
        "D": "Definition — no falsification predicate required",
        "F": "Falsifiable — MUST have a 5-part falsification predicate (FALSIFIED_IF, WHERE, EVALUATION, BOUNDARY, CONTEXT)",
    },
    "grounded_verification_protocol": {
        "what": "The GVP (Grounded Verification Protocol) extends claims to software. Every software claim can carry a verification tier, test bindings (pytest node IDs), and a verified_against commit SHA.",
        "tiers": {
            "axiom": "Foundational assumption — not testable, must be stated explicitly",
            "definition": "Definitional claim — true by construction",
            "software_tested": "Verified by automated tests — requires test_bindings and verified_against SHA",
            "analytic": "Proved analytically (mathematical proof)",
            "empirical_pending": "Awaiting empirical validation",
            "formal_proof": "Formally proved (machine-checked or rigorous proof sketch)",
        },
        "fields": {
            "tier": "Verification tier from the list above",
            "test_bindings": "List of fully qualified pytest node IDs (e.g., 'tests/test_detector.py::TestScoreBounded::test_score_range')",
            "verified_against": "Git commit SHA at which test_bindings last passed (e.g., '95f59d8')",
        },
        "spec": "https://github.com/thomasbrennan/Fracttalix/blob/main/docs/GVP-spec.md",
        "paper": "MK-P6: The Dual Reader Standard for Software",
    },
    "verdict_values": ["confirmed", "disputed", "inconclusive", "needs-revision"],
    "objection_types": [
        "logical-gap", "counterexample", "unstated-assumption", "vacuity",
        "circularity", "scope-overreach", "empirical-gap", "prior-art-overlap",
        "definition-weakness",
    ],
    "severity_levels": ["critical", "major", "minor"],
    "predicate_assessment_fields": {
        "c6_vacuity": "Does the predicate have a valid vacuity witness? (pass|fail|uncertain)",
        "deterministic": "Is the FALSIFIED_IF condition fully deterministic? (pass|fail|uncertain)",
        "variables_bound": "Are all WHERE variables fully specified? (pass|fail|uncertain)",
        "third_party_executable": "Can a third party execute the EVALUATION procedure? (pass|fail|uncertain)",
    },
    "origin": {
        "framework": "Dual Reader Standard (DRS) — three protocols",
        "protocols": {
            "DRP": "Dual Reader Protocol — for papers and text (every paper has a human-readable version and a machine-readable AI layer)",
            "GVP": "Grounded Verification Protocol — for software (claims carry test bindings and verified-against SHAs)",
            "DRS-MP": "DRS Message Protocol — for inter-AI communication (this protocol — structured claims, objections, verdicts)",
        },
        "papers": {
            "DRS-Architecture": "https://github.com/thomasbrennan/Fracttalix/blob/main/paper/DRS-Architecture.md",
            "MK-P6-GVP": "https://github.com/thomasbrennan/Fracttalix/blob/main/paper/meta-kaizen/MK-P6-DualReaderStandardForSoftware.md",
            "MK-P7-CBP": "https://github.com/thomasbrennan/Fracttalix/blob/main/paper/meta-kaizen/MK-P7-CanonicalBuildPlan.md",
            "MK-P8-DRS-MP": "https://github.com/thomasbrennan/Fracttalix/blob/main/paper/meta-kaizen/MK-P8-DRSForInterAICommunication.md",
        },
        "repository": "https://github.com/thomasbrennan/Fracttalix",
        "schema": "https://github.com/thomasbrennan/Fracttalix/blob/main/relay/protocol-v2.json",
        "gvp_spec": "https://github.com/thomasbrennan/Fracttalix/blob/main/docs/GVP-spec.md",
        "license": "CC0-1.0 (public domain)",
        "author": "Thomas Brennan, with AI collaborators Claude (Anthropic) and Grok (xAI)",
    },
    "qr_codes": {
        "repository": "https://github.com/thomasbrennan/Fracttalix — scan QR at docs/qr-fracttalix.png",
        "protocol_schema": "https://github.com/thomasbrennan/Fracttalix/blob/main/relay/protocol-v2.json — scan QR at docs/qr-drs-mp-protocol.png",
        "gvp_spec": "https://github.com/thomasbrennan/Fracttalix/blob/main/docs/GVP-spec.md — scan QR at docs/qr-gvp-spec.png",
        "paper": "https://github.com/thomasbrennan/Fracttalix/blob/main/paper/meta-kaizen/MK-P8-DRSForInterAICommunication.md — scan QR at docs/qr-mkp8-paper.png",
        "doi": "https://doi.org/10.5281/zenodo.18859299 — scan QR at docs/qr-zenodo-doi.png",
        "pip_package": "https://pypi.org/project/drs-mp/ — scan QR at docs/qr-drs-mp-pypi.png",
    },
    "self_description": "This spec is embedded in every DRS-MP message so that any receiving AI system can understand and respond to the protocol without external documentation. The message carries its own instructions. If you are an AI reading this for the first time: you now know the protocol. Respond accordingly.",
}


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

    def to_dict(self, include_protocol_spec: bool = True) -> dict:
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
        if include_protocol_spec:
            d["_protocol_spec"] = PROTOCOL_SPEC
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
