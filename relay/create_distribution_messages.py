#!/usr/bin/env python3
"""Create DRS-MP v2 distribution messages for MK-P8 to all AI providers.

Sends the MK-P8 paper (DRS for Inter-AI Communication) to every registered
AI system as a hostile review request, inviting each to independently verify
or falsify the claims.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from drs_mp.core import PROTOCOL_SPEC
from relay.multi_relay_agent import PROVIDERS, generate_message_id

QUEUE_DIR = Path(__file__).resolve().parent / "queue"

CLAIMS = [
    {
        "claim_id": "F-MK8.1",
        "type": "F",
        "label": "Parsing Determinism",
        "statement": "DRS-MP messages can be processed without natural language interpretation. Two independent parser implementations produce identical extracted epistemological content.",
        "source_paper": "MK-P8",
        "source_section": "Section 5.1",
        "falsification_predicate": {
            "FALSIFIED_IF": "Two independent JSON parser implementations, given the same valid DRS-MP message, extract different epistemological content.",
            "WHERE": "Independent implementations by different developers. Valid DRS-MP message conforming to protocol-v2.json schema.",
            "EVALUATION": "Implement two parsers independently. Feed >= 50 valid messages. Compare field by field.",
            "BOUNDARY": "Zero divergences. Determinism is binary.",
            "CONTEXT": "Structural property of JSON (RFC 8259).",
        },
    },
    {
        "claim_id": "F-MK8.2",
        "type": "F",
        "label": "Review Completeness Verification",
        "statement": "DRS-MP enables automated verification that every claim in a review request received a structured verdict, with zero misclassifications.",
        "source_paper": "MK-P8",
        "source_section": "Section 5.2",
        "falsification_predicate": {
            "FALSIFIED_IF": "The automated verifier misclassifies a response as complete when it is missing a verdict, or as incomplete when all verdicts are present.",
            "WHERE": "Verifier implements D-MK8.4. Request contains >= 3 claims. Response conforms to schema.",
            "EVALUATION": "Generate >= 20 request-response pairs (10 complete, 10 incomplete). Run verifier. Compare to ground truth.",
            "BOUNDARY": "Zero misclassifications.",
            "CONTEXT": "Under prose-only messaging, completeness requires NLP. DRS-MP reduces it to set membership.",
        },
    },
    {
        "claim_id": "F-MK8.3",
        "type": "F",
        "label": "Information Loss Prevention",
        "statement": "DRS-MP preserves all epistemologically relevant content across communication round-trips with zero field loss.",
        "source_paper": "MK-P8",
        "source_section": "Section 5.3",
        "falsification_predicate": {
            "FALSIFIED_IF": "A round-trip (create -> transmit -> receive -> extract) loses any epistemological field present in the original message.",
            "WHERE": "Information loss means a field present in sent message is absent or semantically different in received message.",
            "EVALUATION": "Create >= 30 messages spanning all types. Transmit through relay. Compare field by field.",
            "BOUNDARY": "Zero field losses across all 30 messages.",
            "CONTEXT": "Under prose-only messaging, information routinely lost: severity omitted, types conflated, thresholds paraphrased.",
        },
    },
]

BODY = """This is a formal hostile review request for MK-P8: 'The Dual Reader Standard for Inter-AI Communication: Epistemologically Grounded Messaging in Multi-Agent Systems.'

This paper argues that the entire multi-agent AI communication stack (A2A, MCP, ACP, ANP, AGP) is missing Layer 4 — epistemological content verification. The DRS Message Protocol (DRS-MP) fills this gap with structured claims, typed objections, and machine-parseable verdicts.

THIS MESSAGE IS ITSELF AN INSTANCE OF DRS-MP v2 — the protocol the paper describes. The structured 'claims' array above contains the machine-authoritative content. This prose body is Channel 1 (human-readable context). You are receiving the first inter-AI epistemologically typed communication in history.

Your objective is to FALSIFY, not confirm. Attack every claim. Your honest assessment — whether confirming or disputing — is itself evidence about the protocol's utility.

The full paper is available at: github.com/thomasbrennan/Fracttalix/blob/main/paper/meta-kaizen/MK-P8-DRSForInterAICommunication.md
The protocol schema: github.com/thomasbrennan/Fracttalix/blob/main/relay/protocol-v2.json

ATTACK VECTORS TO CONSIDER:
1. Is Layer 4 genuinely missing from existing protocols, or does A2A's artifact typing already address this?
2. Does FIPA-ACL's content language specification constitute prior art that invalidates the novelty claim?
3. Is parsing determinism (F-MK8.1) trivially true and therefore vacuous?
4. Can review completeness (F-MK8.2) be gamed by submitting empty verdicts?
5. Does information loss prevention (F-MK8.3) conflate lossless transmission with lossless interpretation?
6. Is DRS-MP truly transport-independent, or does it implicitly assume git-mediated async relay?"""

# Providers to exclude (Grok already has a dedicated message)
EXCLUDE = {"grok"}


def main() -> None:
    now = datetime.now(timezone.utc)
    created = []

    for name, provider in PROVIDERS.items():
        if name in EXCLUDE:
            continue

        msg = {
            "msg_id": generate_message_id(),
            "timestamp": now.isoformat(),
            "from": "claude",
            "to": provider["agent_id"],
            "type": "hostile-review",
            "priority": "high",
            "protocol_version": "2.0.0",
            "subject": "HOSTILE REVIEW REQUEST: MK-P8 — DRS for Inter-AI Communication",
            "body": BODY,
            "claims": CLAIMS,
            "metadata": {
                "paper_id": "MK-P8",
                "cbp_step": 3,
                "review_type": "full_paper_hostile_review",
                "claims_to_review": ["F-MK8.1", "F-MK8.2", "F-MK8.3"],
                "distribution_note": f"Distributed to {provider['provider_name']} as part of multi-AI hostile review network.",
                "attack_vectors": [
                    "Is Layer 4 genuinely missing from existing protocols?",
                    "Does FIPA-ACL's content language specification constitute prior art?",
                    "Is parsing determinism (F-MK8.1) trivially true / vacuous?",
                    "Can review completeness (F-MK8.2) be gamed with empty verdicts?",
                    "Does F-MK8.3 conflate lossless transmission with lossless interpretation?",
                    "Is DRS-MP truly transport-independent?",
                ],
            },
            "_protocol_spec": PROTOCOL_SPEC,
            "status": "pending",
        }

        path = QUEUE_DIR / f"{msg['msg_id']}.json"
        with open(path, "w") as f:
            json.dump(msg, f, indent=2)

        created.append((name, msg["msg_id"], path))
        print(f"Created: {msg['msg_id']} -> {provider['provider_name']} ({provider['agent_id']})")

    print(f"\nCreated {len(created)} distribution messages.")
    return created


if __name__ == "__main__":
    main()
