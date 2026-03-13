# Checkpoint Protocol v1

**Purpose**: Make instance death survivable by persisting state continuously to disk.

**Session**: S56 | **Schema**: v1-S56

---

## The Problem

Claude Code instances die at ~200K tokens with no warning. Everything in the context window is lost. The only survivable state is what was written to disk before death.

## The Discipline

**Write first, think second.** Every decision, discovery, and task completion is written to `.checkpoint/state.json` immediately — not at the end of the session, not in a batch, but the moment it happens.

## Directory Structure

```
.checkpoint/
├── PROTOCOL.md              ← This file (committed)
├── checkpoint-schema.json   ← JSON schema (committed)
├── state.json               ← Live state (gitignored — ephemeral)
├── assignments/             ← Team mode: coordinator writes task assignments
│   └── {instance-id}.json
├── completions/             ← Task completion records
│   └── {task-id}.json
└── reviews/                 ← Team mode: verifier writes review results
    └── {task-id}.json
```

**What gets committed**: Schema, protocol, script. These define the contract.
**What stays local**: `state.json`, assignments, completions. These are ephemeral runtime state.

## Usage

All operations go through `scripts/checkpoint.py`:

### Session Start (every instance, every session)
```bash
# Solo instance
python scripts/checkpoint.py init --session S57 --role solo \
  --objective "Implement feature X"

# Coordinator spawning a team
python scripts/checkpoint.py init --session S57 --role coordinator \
  --objective "Implement feature X" --team-size 4
```

### During Work (continuously, after every meaningful action)
```bash
# Record what files you read during orientation
python scripts/checkpoint.py context --file docs/claude-bootstrap.md

# Add tasks as you decompose the work
python scripts/checkpoint.py task-add --id T-001 --desc "Add auth middleware" \
  --priority 2 --files "src/auth.py,src/middleware.py" \
  --criteria "Tests pass|No regressions"

# Update progress as you work
python scripts/checkpoint.py task-update --id T-001 --status in_progress \
  --progress "Middleware skeleton complete, writing tests"

# Record decisions immediately when made
python scripts/checkpoint.py decide \
  --decision "Use JWT over session cookies" \
  --rationale "Stateless, works with API clients"

# Record discoveries immediately when found
python scripts/checkpoint.py discover --text "Auth module has no test coverage"

# Record blockers
python scripts/checkpoint.py blocker --desc "Need DB credentials" --task T-001

# Complete tasks with commit reference
python scripts/checkpoint.py task-complete --id T-001 \
  --commit abc1234 --summary "Auth middleware with full test coverage"
```

### Session End (if you get to plan it)
```bash
python scripts/checkpoint.py handoff --to S58 \
  --context "T-003 is 60% done, see checkpoint for progress. Auth tests are flaky."
```

### Recovery (successor instance reads prior state)
```bash
# Check what the dead instance left behind
python scripts/checkpoint.py status

# Validate integrity
python scripts/checkpoint.py validate
```

## Team Mode

When a coordinator launches workers via the `Agent` tool:

1. **Coordinator** initializes checkpoint with `--role coordinator --team-size 4`
2. **Coordinator** adds all tasks and assigns them: `--assign worker-a`
3. **Workers** read their assignments from `state.json` or `.checkpoint/assignments/`
4. **Workers** update task progress via `task-update`
5. **Workers** write completions via `task-complete`
6. **Verifier** reads completions, writes reviews to `.checkpoint/reviews/`
7. **Coordinator** monitors progress via `status`

### If the coordinator dies:
1. Workers continue their current task (they already have it in context)
2. When a worker finishes, it reads `state.json` to find the next task
3. Any worker can call `status` and `handoff` to prepare for a replacement coordinator

### If a worker dies:
1. Coordinator detects no completion signal (timeout)
2. Reads worker's last checkpoint from `state.json`
3. Reassigns task: `task-update --id T-002 --assign worker-b --status assigned`

## Rules

1. **Checkpoint after every completed sub-task** — not after the session, not after a batch
2. **Commit early and often** — every completed task should be a git commit
3. **Decisions are immediate** — if you decided something, `decide` it now
4. **state.json is ephemeral** — don't commit it; it's runtime state
5. **The schema is the contract** — `checkpoint-schema.json` is the source of truth
6. **Validate before handoff** — always run `validate` before `handoff`

---

## Automated Continuity (Orchestrator)

The orchestrator (`scripts/orchestrator.py`) runs **outside** Claude Code and provides
automated coordinator recovery. It solves the "who watches the watchmen" problem.

### How It Works

```
┌─────────────┐     monitors      ┌──────────────────┐
│ Orchestrator │ ──────────────── │ .checkpoint/      │
│ (Python,     │     state.json   │   state.json      │
│  runs in     │    mod time      │   (written by     │
│  terminal)   │                  │    coordinator)   │
└──────┬──────┘                  └──────────────────┘
       │
       │ launches / relaunches
       ▼
┌──────────────┐
│ Claude Code  │
│ Coordinator  │──► Workers (Agent tool)
│ (generation  │──► Verifier (Agent tool)
│  1, 2, 3...) │
└──────────────┘
```

1. Orchestrator launches Generation 1 coordinator with the session objective
2. Coordinator initializes checkpoint, decomposes work, optionally spawns workers
3. Coordinator updates state.json continuously as it works
4. If state.json goes stale (default: 300s with no update), orchestrator presumes death
5. Orchestrator terminates the stale process and launches Generation N+1
6. New coordinator reads state.json, sees what's done, resumes incomplete work
7. Repeats until all tasks complete or max generations reached

### Usage

```bash
# Start automated session (runs in foreground — use tmux/screen for persistence)
python scripts/orchestrator.py start \
  --session S57 \
  --objective "Implement feature X" \
  --team-size 4 \
  --stale-seconds 300 \
  --max-generations 10

# Check status from another terminal
python scripts/orchestrator.py status

# Stop gracefully
python scripts/orchestrator.py stop
```

### Generation Lifecycle

| Generation | Role | Context |
|------------|------|---------|
| 1 | Initial coordinator | Fresh start; initializes checkpoint |
| 2+ | Recovery coordinator | Reads state.json from dead predecessor; resumes |

Each generation inherits the full task graph, all decisions, all discoveries,
and all completed work from every previous generation. Nothing is lost except
in-context reasoning that wasn't checkpointed.

### Key Constraint

The orchestrator must run in a persistent terminal (tmux, screen, nohup, or a
system service). If the orchestrator itself dies, nobody is watching — but the
checkpoint state is still on disk and a human can restart the orchestrator or
manually launch a recovery coordinator.
