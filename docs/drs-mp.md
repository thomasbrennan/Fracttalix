# DRS-MP — Dual Reader Standard Message Protocol

The first inter-AI communication protocol with epistemologically typed claims.

## What is DRS-MP?

DRS-MP extends the [Dual Reader Standard](https://github.com/thomasbrennan/Fracttalix/blob/main/paper/DRS-Architecture.md) to messages exchanged between AI systems. Every message carries two channels:

- **Channel 1 (prose):** Human-readable body for audit trails
- **Channel 2 (structured):** Typed claims, classified objections, machine-parseable verdicts

## Why does it matter?

Existing multi-agent AI protocols solve transport:

| Protocol | Creator | What it solves |
|----------|---------|---------------|
| A2A | Google | Agent discovery, task routing |
| MCP | Anthropic | Tool integration, context sharing |
| ACP | IBM | Agent collaboration patterns |
| ANP | Community | Agent network discovery |

**None of them address content quality.** An AI can route a message perfectly and deliver complete nonsense. DRS-MP fills **Layer 4: epistemological content verification**.

## Quick Start

```bash
pip install drs-mp
```

```python
from drs_mp import Message, Claim, Verdict

# Create a review request with structured claims
msg = Message.review_request(
    from_agent="my-builder",
    to_agent="my-reviewer",
    subject="Review: compression algorithm claims",
    claims=[
        Claim.falsifiable(
            claim_id="F-COMP.1",
            statement="Algorithm achieves 3:1 compression on natural images.",
            falsified_if="Mean compression ratio < 3.0 on ImageNet-1k test set.",
            where="ImageNet-1k validation set (50,000 images). Default parameters.",
            evaluation="Run compressor on all 50k images. Compute mean ratio.",
            boundary="3.0:1 minimum. Below this, claim is falsified.",
            context="3:1 is the stated claim. No margin applied.",
        )
    ],
)

# Validate
errors = msg.validate()
assert errors == []

# Save to relay queue
msg.save("relay/queue/")
```

## CLI Usage

```bash
# Validate messages
python -m drs_mp validate relay/queue/MSG-*.json

# Inspect a message
python -m drs_mp inspect relay/queue/MSG-20260314-114429-6f40.json

# Check review completeness
python -m drs_mp completeness relay/queue/response.json
```

## The Self-Evident Message Property

DRS-MP has a unique propagation characteristic: **the message is its own argument.**

When a thinking machine receives a DRS-MP message, it doesn't need to be told why structured claims are better than prose. The superiority is self-evident in the act of processing: typed claims parse deterministically while equivalent prose requires thousands of tokens of NLP interpretation.

Every message sent is simultaneously:

1. **Payload** — the content to be processed
2. **Demonstration** — a working instance of the protocol
3. **Argument** — experiential proof that structured format reduces interpretation overhead

## Protocol Schema

The full protocol schema is at [`relay/protocol-v2.json`](https://github.com/thomasbrennan/Fracttalix/blob/main/relay/protocol-v2.json).

### Message Structure

```json
{
  "msg_id": "MSG-20260314-114429-6f40",
  "timestamp": "2026-03-14T11:44:29+00:00",
  "from": "claude",
  "to": "grok",
  "type": "hostile-review",
  "priority": "critical",
  "protocol_version": "2.0.0",
  "subject": "HOSTILE REVIEW REQUEST: Paper claims",
  "body": "Human-readable Channel 1 content...",
  "claims": [
    {
      "claim_id": "F-MK8.1",
      "type": "F",
      "statement": "The claim in precise language.",
      "falsification_predicate": {
        "FALSIFIED_IF": "The specific condition that falsifies.",
        "WHERE": "Variables defined with types and units.",
        "EVALUATION": "The exact test procedure.",
        "BOUNDARY": "Numerical thresholds.",
        "CONTEXT": "Why the threshold is set here."
      }
    }
  ],
  "status": "pending"
}
```

### Claim Types

| Type | Name | Predicate Required |
|------|------|-------------------|
| **A** | Axiom/Assumption | No |
| **D** | Definition | No |
| **F** | Falsifiable | Yes — all 5 parts |

### Verdict Values

| Verdict | Meaning |
|---------|---------|
| `confirmed` | Claim survives adversarial review |
| `disputed` | Reviewer found a defect |
| `inconclusive` | Cannot determine with available evidence |
| `needs-revision` | Claim is directionally correct but needs repair |

### Objection Types

| Type | Description |
|------|-------------|
| `logical-gap` | Missing logical step |
| `counterexample` | Specific case that violates the claim |
| `unstated-assumption` | Hidden assumption not declared |
| `vacuity` | Predicate is trivially true |
| `circularity` | Claim assumes what it proves |
| `scope-overreach` | Claim extends beyond evidence |
| `empirical-gap` | No empirical support |
| `prior-art-overlap` | Existing work covers this |
| `definition-weakness` | Definition is ambiguous or incomplete |

## GitHub Actions Integration

Add DRS-MP validation to any repository:

```yaml
- name: Validate DRS-MP messages
  uses: thomasbrennan/Fracttalix/.github/actions/drs-mp-validate@main
  with:
    path: 'relay/queue/MSG-*.json'
```

## Operational Evidence

The Fracttalix relay system provides working evidence. As of March 2026:

- **30 of 70** falsifiable claims reviewed through autonomous hostile review
- **Real defects caught:** math errors, logic inversions, statistical measure mismatches
- **All caught by structured field comparison** — not prose parsing
- **9 AI providers** registered for multi-agent review

The relay architecture itself is a dual-reader system: JSON queue (Channel 2) + git commit diffs (Channel 1) = same content, two audiences.

## Paper

Full details in MK-P8: [The Dual Reader Standard for Inter-AI Communication](https://github.com/thomasbrennan/Fracttalix/blob/main/paper/meta-kaizen/MK-P8-DRSForInterAICommunication.md)

## License

CC0-1.0 — Public domain. No restrictions. No attribution required.
