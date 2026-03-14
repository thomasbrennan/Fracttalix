# Quality Control Cross-Referencing Protocol

> How Claude and Grok cross-check each other's work through the relay.

## Overview

The QC protocol uses the multi-AI relay to enable independent verification.
Neither AI sees the other's reasoning until after committing their own assessment.
This prevents confirmation bias and maximizes the value of having two independent models.

## Workflow

### Standard flow (human-relayed)

1. **Claude authors** a claim or change, commits to repo
2. **Claude creates a QC request** via `python -m relay.relay_manager qc-review --to grok ...`
3. **Thomas relays to Grok** — copies message into a Grok session (bootstrapped with `relay/grok-bootstrap.md`)
4. **Grok reviews** and produces a response with verdict/confidence/reasoning
5. **Thomas commits** Grok's response JSON to `relay/queue/`
6. **Claude reads** the response and addresses feedback

### Autonomous flow (AI-initiated, no human relay required)

Either AI can initiate communication independently:

1. **AI detects something** — discrepancy, relevant finding, completed milestone, question
2. **AI creates a message** in `relay/queue/` with appropriate type and priority
3. **Message is committed and pushed** (Claude does this directly; Grok asks Thomas or uses repo access)
4. **Recipient picks up** the message at session start or during work
5. **Recipient responds** through the relay

### When AIs should auto-initiate

| Trigger | Initiator | Message Type | Priority |
|---------|-----------|-------------|----------|
| Claim authored | Either | `claim-review` | normal |
| Inconsistency found | Either | `cross-reference` | high |
| External literature found | Grok | `cross-reference` | normal |
| Code review concern | Either | `qc-request` | normal |
| Milestone reached | Either | `general` | low |
| Blocking question | Either | `status-query` | high |

### Guardrails

- **Max 10 messages per session** per AI
- **Critical priority requires human approval** before sending
- **Claim modifications require human approval** — review is autonomous, changes are not
- **Thomas is CC'd on all messages** — can override, veto, or add context at any time
- **No message loops** — if both AIs are responding to each other, cap at 3 exchanges then escalate to Thomas

## QC Categories

### Claim Verification (`claim-review`)
- Check falsification predicate logical consistency
- Verify mathematical derivations against known results
- Confirm test bindings actually test what they claim
- Cross-reference with relevant literature

### Code Review (`qc-request`)
- Check for bugs, edge cases, and security issues
- Verify consistency with existing codebase patterns
- Confirm tests adequately cover the change
- Review for maintainability

### Fact-Checking (`cross-reference`)
- Verify empirical claims against databases
- Check citations and references
- Confirm no prior art conflicts
- Validate scope boundary claims

## Escalation

If both AIs disagree with `confidence > 0.7`:
1. The disagreement is flagged to Thomas
2. Both reasoning chains are preserved in the archive
3. Thomas makes the final determination
4. The resolution is documented in the relevant AI layer

## Metrics

The relay tracks:
- **Messages sent/received** per agent
- **Verdict distribution** (confirmed/disputed/inconclusive/needs-revision)
- **Average confidence** per agent per type
- **Time to resolution** (commit timestamps)

These are computable from the JSONL archive and git history.

## Anti-Patterns

- **Rubber-stamping**: Responding "confirmed" without actual verification
- **Scope creep**: Reviewing things not asked about
- **Circular reasoning**: Using the other AI's output as evidence
- **Stale reviews**: Reviewing against outdated code (always check HEAD)
