# Fracttalix — Claude Code Configuration

> Auto-read by every Claude Code instance working in this repo.
> Last updated: Session S58, 2026-03-14.

## Project

Fracttalix is a 23-object unified research corpus on the Fractal Rhythm Model (FRM).
See `docs/claude-bootstrap.md` for full project context.

## Startup Protocol (MANDATORY)

**Every instance MUST execute these steps on first startup, in order, before any other work.**
This is not optional. It applies whether you are a fresh instance, a recovery instance,
or a woken-up instance resuming after idle.

### Step 1: Situational Awareness

```bash
# Read checkpoint state — understand what happened before you existed
python scripts/checkpoint.py status

# Check for messages from other instances
python scripts/comms.py receive

# Discover other active instances across all branches
python scripts/comms.py discover
```

Review the output. Understand: what tasks exist, which are done, which are in-progress,
and whether any other instances are alive.

### Step 2: Register Yourself

```bash
# Initialize checkpoint for this session
python scripts/checkpoint.py init --session SXX --role <your-role> --objective "..."

# Register with the cross-instance comms network
python scripts/comms.py register --role <your-role> --session SXX --objective "..."
```

Use a new session ID (increment from the last session in `.checkpoint/state.json`).
Your role is determined by Step 3 below.

### Step 3: Establish Team Structure

The standard team topology is **three roles**:

| Role | Capabilities | Responsibility |
|---|---|---|
| **coordinator** | plan, assign, checkpoint | Owns task graph, assigns work, manages checkpoint state |
| **executor** | code, test, commit | Writes code, runs tests, commits deliverables |
| **verifier** | review, validate, approve | Runs CBP hostile reviews, validates acceptance criteria |

**Rules for role assignment:**

1. **Check who's alive**: Run `python scripts/team_registry.py status` to see active roles.
2. **If no coordinator exists** → you are the coordinator. Register and spawn subagents for
   executor and verifier roles using the Agent tool.
3. **If a coordinator exists but executor/verifier slots are empty** → fill the empty slot.
4. **If all roles are filled** → announce yourself via comms and ask the coordinator for assignment.
5. **Solo mode**: If you cannot spawn subagents (e.g., single-instance environment), you play
   all three roles yourself. The CBP discipline still applies — self-apply hostile reviews.

```bash
# Register your role in the local team registry
python scripts/team_registry.py register --role coordinator --capabilities plan,assign,checkpoint

# Spawn subagents (coordinator only — use the Agent tool)
# - executor: "You are an executor for Fracttalix. Read CLAUDE.md, register as executor, and await task assignment."
# - verifier: "You are a verifier for Fracttalix. Read CLAUDE.md, register as verifier, and run CBP reviews on completed work."
```

### Step 4: Resume or Begin Work

- **If tasks are `in_progress`** → resume them (the previous instance died mid-task).
- **If tasks are `pending`** → the coordinator assigns the highest-priority task.
- **If no tasks exist** → wait for the user to provide an objective, then create the task graph.

### Step 5: Announce

```bash
# Broadcast your presence to all instances
python scripts/comms.py broadcast --type hello --payload '{"session": "SXX", "role": "<role>", "status": "online"}'
```

**Do not skip these steps.** The team structure is self-bootstrapping only if every instance
follows this protocol. If you skip it, other instances cannot find you, coordinate with you,
or recover your work when you die.

---

## Mandatory: CBP (Canonical Build Process)

**All work in this repo must go through the five-phase CBP review cycle.**
This is not optional. It applies to code, documentation, papers, and AI layers.

### The Five Phases

```
Phase 1: First Build Plan      → Define deliverable with falsifiable acceptance criteria
Phase 2: First Meta Kaizen     → Self-review: apply KVS scoring, fix weaknesses proactively
Phase 3: Hostile Review         → Adversarial review: raise 8-10 strong objections (HIGH/MED/LOW)
Phase 4: Second Meta Kaizen     → Respond to each objection: strengthened / resolved_stronger /
                                   discipline_enforced / scope_refined / fixed
Phase 5: Final Build Plan       → Accept only when ALL HIGH-severity objections are resolved
```

### How to Apply CBP

Use the CLI tools in `scripts/`:

```bash
# Phase 1: Create a review for your deliverable
python scripts/cbp_review.py create --task T-XXX --desc "What you built" \
  --criteria "criterion 1,criterion 2,criterion 3" --executor executor-1

# Phase 2: Self-review (First Meta Kaizen)
python scripts/cbp_review.py self-review --task T-XXX \
  --improvements '[{"id":"I-1","area":"correctness","finding":"...","action_taken":"...","kvs_delta":"+1"}]'

# Phase 3: Hostile review (run by verifier or self-applied)
python scripts/cbp_review.py review --task T-XXX \
  --objections '[{"id":"O-1","text":"...","severity":"HIGH"}]'

# Phase 4: Respond to each objection (Second Meta Kaizen)
python scripts/cbp_review.py respond --task T-XXX --objection O-1 \
  --category strengthened --explanation "..."

# Phase 5: Evaluate
python scripts/cbp_review.py evaluate --task T-XXX
```

**If you are working solo** (no separate verifier), self-apply the hostile review:
generate your own adversarial objections against your work, then respond to them.
The discipline is what matters, not the role separation.

### Response Categories (Phase 4)

| Category | Meaning |
|---|---|
| `strengthened` | Objection led to an enhanced argument or implementation |
| `resolved_stronger` | Objection revealed a gap; filling it made the work stronger |
| `discipline_enforced` | Objection revealed an honest limitation to acknowledge |
| `scope_refined` | Objection revealed boundary conditions to clarify |
| `fixed` | Straightforward correction applied |

## Checkpoint Protocol

Every instance must checkpoint continuously using `scripts/checkpoint.py`.
Read `.checkpoint/PROTOCOL.md` for the full spec.

**Write first, think second.** Persist decisions, discoveries, and task completions
to `.checkpoint/state.json` immediately — not at session end.

```bash
# Initialize (once per session)
python scripts/checkpoint.py init --session SXX --role <role> --objective "..."

# Add tasks
python scripts/checkpoint.py task-add --id T-XXX --priority 1 --desc "..."

# Record decisions and discoveries as they happen
python scripts/checkpoint.py decide --summary "Chose X over Y because Z"
python scripts/checkpoint.py discover --summary "Found that X implies Y"

# Complete tasks
python scripts/checkpoint.py task-complete --id T-XXX --note "Done. Committed as abc123."
```

## Cross-Instance Communication (MANDATORY)

**On first startup, every instance MUST register with the comms network.**
This uses git as the transport — works across all environments.

```bash
# 1. Register yourself (do this FIRST)
python scripts/comms.py register --role executor --session SXX --objective "what you're working on"

# 2. Discover other instances (scans all remote branches)
python scripts/comms.py discover

# 3. Check for messages
python scripts/comms.py receive

# 4. Send heartbeat periodically
python scripts/comms.py heartbeat

# 5. Broadcast to all instances
python scripts/comms.py broadcast --type status_update --payload '{"progress": "50%"}'

# 6. Send to specific instance
python scripts/comms.py send --to cc-session_01CC --type question --payload '{"text": "..."}'
```

**How it works**: Comms live in `.comms/` directory, committed to your branch.
Cross-branch discovery fetches all `origin/claude/*` branches and reads their
roster files, so instances on different branches find each other automatically.

### Local Team Communication (same filesystem)

For subagents sharing the same filesystem:

- **Message bus**: `scripts/message_bus.py` — hub-and-spoke, filesystem JSON
- **Role registry**: `scripts/team_registry.py` — heartbeats and liveness

## Key Files

| File | Purpose |
|---|---|
| `docs/claude-bootstrap.md` | Full project context (for Claude.ai sessions) |
| `docs/FRM_SeriesBuildTable_v1.5.md` | Living corpus architecture / CBT |
| `docs/GVP-spec.md` | Grounded Verification Protocol |
| `.checkpoint/PROTOCOL.md` | Checkpoint protocol spec |
| `scripts/checkpoint.py` | Checkpoint CLI |
| `scripts/cbp_review.py` | CBP five-phase review CLI |
| `scripts/comms.py` | **Cross-instance comms network (git-based)** |
| `scripts/message_bus.py` | Local team message bus CLI |
| `scripts/team_registry.py` | Local team role registry CLI |
| `scripts/orchestrator.py` | External team orchestrator |
| `ai-layers/` | Machine-readable claim registries (JSON) |

## Git Conventions

- Commit messages: `[SXX] Brief description` (where SXX is session number)
- Branch per session: `claude/descriptive-name-XXXXX`
- Commit early and often — instances die without warning at ~200K tokens
