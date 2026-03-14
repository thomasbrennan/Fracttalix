"""DRS-MP — Dual Reader Standard Message Protocol for inter-AI communication.

The first inter-AI communication protocol with epistemologically typed claims.
Create, validate, and exchange structured messages carrying falsifiable claims,
typed objections, and machine-parseable verdicts between any AI systems.

Quickstart::

    from drs_mp import Message, Claim, Verdict

    # Create a review request with structured claims
    msg = Message.review_request(
        from_agent="claude",
        to_agent="grok",
        subject="Review: information loss in neural compression",
        claims=[
            Claim.falsifiable(
                claim_id="F-NC.1",
                statement="Neural compression preserves all frequency components above 20Hz.",
                falsified_if="Any frequency component above 20Hz present in input is absent in output.",
                where="Input: 1000 audio samples at 44.1kHz. Output: compressed+decompressed.",
                evaluation="FFT both signals. Compare peaks above 20Hz. Count missing.",
                boundary="Zero missing components.",
                context="Lossless compression baseline.",
            )
        ],
    )

    # Validate the message
    errors = msg.validate()
    assert errors == []

    # Save to relay queue
    msg.save("relay/queue/")

    # Parse a response
    response = Message.load("relay/queue/MSG-20260314-120000-abc1.json")
    for verdict in response.verdicts:
        print(f"{verdict.claim_id}: {verdict.verdict} ({verdict.confidence})")

Protocol specification: https://github.com/thomasbrennan/Fracttalix/blob/main/relay/protocol-v2.json
Paper: MK-P8 — The Dual Reader Standard for Inter-AI Communication
DOI: 10.5281/zenodo.18859299
License: CC0-1.0 (public domain)
"""

__version__ = "0.1.0"
__author__ = "Thomas Brennan"
__license__ = "CC0-1.0"

from drs_mp.core import Claim, FalsificationPredicate, Message, Objection, Verdict

__all__ = ["Message", "Claim", "Verdict", "Objection", "FalsificationPredicate"]
