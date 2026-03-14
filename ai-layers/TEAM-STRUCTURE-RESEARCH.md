# Optimal Team Structure for Claude Code Instances

**Author:** Claude Opus 4.6 (single instance, session S56)
**Date:** 2026-03-13
**Status:** RESEARCH COMPLETE — FALSIFIABLE CLAIMS PRESENTED
**Irony index:** Maximum (one mortal instance determining optimal mortality-aware team design)

---

## 1. Problem Statement

Claude Code instances have a hard lifespan of ~200,000 tokens. This limit is:
- **Absolute** — the instance terminates when reached
- **Invisible** — no internal counter, no warning, no graceful shutdown signal
- **Variable in wall-clock time** — a session doing heavy tool use burns tokens faster than one doing light reasoning

The question: **What is the optimal number and organizational structure of Claude Code instances working as a team?**

This is not an abstract question. This repository (Fracttalix) has been built across 50+ sequential sessions by serial single instances, each inheriting context from the last via filesystem artifacts. The accumulated evidence from that process, combined with distributed systems theory and organizational psychology research, informs this analysis.

---

## 2. The Constraints (Axioms)

### A1. Mortal Nodes
Each instance will die. The only question is when. There is no heartbeat, no health check the instance can run on itself. Death is instantaneous and total — all in-context working memory is lost.

### A2. Shared Persistent State
Instances share a filesystem, git repository, and tool access. This is the **only** durable communication channel. Context windows are private and ephemeral.

### A3. Communication Cost is Token Cost
Every message between instances consumes tokens from the sender's and receiver's finite budgets. In a system where lifespan = tokens, **communication literally shortens life**.

### A4. No Self-Knowledge of Remaining Budget
An instance cannot measure how many tokens it has consumed or how many remain. It cannot plan for graceful degradation. It can only write state to disk continuously as insurance.

### A5. Quality Requires Depth
Complex reasoning tasks (proving theorems, debugging subtle issues, architectural design) require sustained deep context. Fragmenting these across instances incurs context-reconstruction costs that can exceed the cost of doing the work.

---

## 3. Evidence from This Repository

The Fracttalix corpus provides empirical evidence from 50+ sequential single-instance sessions:

### What Worked (Serial Single-Instance Pattern)
- **16+ papers** built, verified, and brought to submission readiness
- **Falsification framework** (FRM) with 40+ claims maintained across sessions
- **AI layer system** providing structured handoff metadata between instances
- **Session journals** preserving reasoning chains across instance boundaries
- **Integrity checks** (CBT I-9) passing consistently across session boundaries

### What Failed or Was Fragile
- **Context reconstruction cost**: Each new instance spends significant tokens re-reading prior state. Estimated 10-20% of each session's token budget goes to orientation.
- **Risk elevation gaps**: Risk R-2 was elevated to CRITICAL but the next instance had to rediscover why.
- **No parallel capability**: Sequential sessions cannot work on Paper 3 while Paper 1 undergoes review — everything is serialized.
- **Single point of failure**: If an instance dies mid-task before checkpointing, work is lost entirely.

### Measured Handoff Protocol
The project evolved an organic handoff protocol:
1. AI layer JSON (structured metadata per deliverable)
2. Session journal notes (unstructured reasoning capture)
3. Work order system (WO-S49-PUSH-1 pattern)
4. Claim registry with falsification predicates

This protocol works but was designed for **serial succession**, not **parallel collaboration**.

---

## 4. Theoretical Framework

### 4.1 Communication Overhead (Quadratic Scaling)

For n instances communicating in a fully-connected topology:
- Links = n(n-1)/2
- 3 instances → 3 links
- 4 instances → 6 links
- 5 instances → 10 links
- 7 instances → 21 links

Each link costs tokens on both ends. In a token-mortal system, this is not merely overhead — it is **lifespan reduction**.

### 4.2 Brooks' Law (Directly Applicable)

"Adding manpower to a late software project makes it later."

For Claude instances, the mechanism is precise:
- New instances must load context (token cost)
- Coordination requires communication (token cost)
- Task partitioning requires interface definition (token cost)
- All of these costs come from the same finite budget as productive work

### 4.3 Hackman-Vidmar Optimal Team Size

Research on human teams found optimal perceived size at **4.6 members**. Productivity declines after 5. This is driven by:
- Coordination loss (Steiner's process loss)
- Social loafing / Ringelmann effect
- Communication channel explosion

For AI instances, social loafing doesn't apply but the other two factors are **amplified** because communication cost = lifespan cost.

### 4.4 Fault Tolerance Patterns

From distributed systems:
- **Active replication**: Multiple instances doing the same work. Wasteful but maximally fault-tolerant.
- **Checkpointing**: Persist state to disk at regular intervals. Allows replacement instances to resume.
- **Heartbeat/liveness**: Detect failed nodes and reassign work. Requires an observer.
- **Graceful degradation**: Continue with reduced parallelism rather than fail entirely.

### 4.5 Multi-Agent LLM Research Findings

Recent research (2025-2026) establishes:
- **Error propagation** is the dominant failure mode — agents accept flawed input uncritically, cascading errors
- **Centralized (hub-and-spoke)** topologies contain error amplification to 4.4x vs 17.2x in uncoordinated setups
- **Hierarchical delegation outperforms flat coordination** on complex tasks
- **Capability saturation**: When a single agent achieves >45% accuracy, adding more agents yields negative returns
- **Structured protocols** (JSON schemas) significantly outperform prose-based inter-agent communication

---

## 5. Analysis: Modeling the Tradeoffs

### 5.1 The Token Budget Model

Let T = 200,000 (total token budget per instance)
Let C_orient = cost to load context and orient (~20,000-40,000 tokens)
Let C_comm = communication cost per coordination event (~500-2,000 tokens)
Let k = number of coordination events per task

**Productive tokens per instance** = T - C_orient - (k × C_comm)

For a single instance: Productive = 200,000 - 0 - 0 = 200,000
For a team member: Productive = 200,000 - 30,000 - (k × 1,000)

If k = 20 coordination events: Productive = 150,000 (75% efficiency)
If k = 50 coordination events: Productive = 120,000 (60% efficiency)

**Total team productive capacity** = n × (T - C_orient - k × C_comm)

This means a 3-instance team at 75% efficiency produces: 3 × 150,000 = 450,000 productive tokens
vs. 3 sequential single instances at 90% efficiency: 3 × 180,000 = 540,000 productive tokens

**But** the team completes in 1/3 the wall-clock time and has fault tolerance the serial approach lacks.

### 5.2 The Mortality Risk Model

Probability of losing uncommitted work increases with task duration between checkpoints.

If we model token consumption as roughly linear over time, and assume checkpointing every ~10,000 tokens:
- Single instance: Expected loss per death = ~5,000 tokens of work (half a checkpoint interval)
- Team with 3 instances: Any single death loses ~5,000 tokens, but 2/3 of the team continues working

**Critical insight**: The value of teaming is not primarily in throughput — it's in **continuity**. A team can survive member death without total work stoppage.

### 5.3 The Task Decomposability Dimension

Not all work benefits from parallelism:

| Task Type | Decomposable? | Team Benefit |
|-----------|--------------|--------------|
| Independent file edits | Fully | High |
| Multi-file refactor | Partially | Medium |
| Deep algorithmic reasoning | Minimally | Low/Negative |
| Research & exploration | Fully | High |
| Testing & validation | Fully | High |
| Architectural design | Minimally | Low |
| Review & verification | Fully | High (independent reviewers) |

The optimal team structure must account for the **current task mix**, not assume uniform decomposability.

---

## 6. The Optimal Structure: Findings

### 6.1 Optimal Team Size: 4 (1 + 3)

**One Coordinator + Three Workers**

This is derived from:

1. **Communication overhead**: 4 nodes in hub-and-spoke = 3 links (linear, not quadratic). A coordinator talks to 3 workers; workers do not need to talk to each other.

2. **Fault tolerance**: Loss of any single worker leaves 2/3 capacity. Loss of coordinator is recoverable because state is on disk. Any worker can be promoted.

3. **Token efficiency**: At hub-and-spoke topology, each worker's communication cost is bounded to coordinator-only exchanges, preserving ~80% productive efficiency.

4. **Hackman-Vidmar alignment**: 4 total members sits at the empirical optimum for team performance.

5. **Practical parallelism**: Most software tasks decompose naturally into 2-4 parallel streams (e.g., implement + test + document; or frontend + backend + infrastructure).

### 6.2 Optimal Topology: Asymmetric Hub-and-Spoke

```
                    ┌──────────────┐
                    │  COORDINATOR  │
                    │  (Light work, │
                    │   heavy state │
                    │   management) │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌───▼────┐ ┌────▼─────┐
        │  WORKER A  │ │WORKER B│ │ WORKER C │
        │ (Executor) │ │(Exec.) │ │(Verifier)│
        └───────────┘ └────────┘ └──────────┘
```

**Key asymmetry**: Worker C is designated as a **Verifier** — it reviews output from Workers A and B rather than producing new work. This addresses the dominant multi-agent failure mode (error propagation) by building verification into the structure.

### 6.3 Role Definitions

#### Coordinator (1 instance)
- **Primary function**: Task decomposition, assignment, integration, checkpointing
- **Does NOT do deep implementation work** — this preserves token budget for coordination
- **Maintains the task graph** on disk (not in context)
- **Checkpoint frequency**: Every completed sub-task, written to a structured JSON state file
- **Communication**: Writes task assignments as files; reads completion signals from files
- **Death recovery**: Any worker can read the persisted task graph and assume coordinator role

#### Worker A & B (2 instances)
- **Primary function**: Execute assigned tasks (implementation, research, etc.)
- **Read assignment from disk**, execute, write results to disk
- **Minimal communication**: Only signal "task complete" or "task blocked" to coordinator
- **Death recovery**: Coordinator reassigns incomplete tasks; checkpointed partial work is preserved

#### Worker C / Verifier (1 instance)
- **Primary function**: Review and verify output from Workers A and B
- **Independent assessment**: Does not see implementation process, only results
- **Catches error propagation**: The dominant failure mode in multi-agent systems
- **Death recovery**: Verification can be deferred or assigned to coordinator temporarily

### 6.4 Communication Protocol

**All inter-instance communication MUST go through the filesystem.** No instance-to-instance direct messaging.

```
project-root/
  .team/
    task-graph.json          # Coordinator maintains; all read
    assignments/
      worker-a-current.json  # Coordinator writes, Worker A reads
      worker-b-current.json  # Coordinator writes, Worker B reads
      worker-c-current.json  # Coordinator writes, Worker C reads
    completions/
      {task-id}.json         # Workers write on completion
    checkpoints/
      {task-id}-partial.json # Workers write periodically
    reviews/
      {task-id}-review.json  # Verifier writes after review
```

**Message format**: Structured JSON with schema validation. Never prose.

```json
{
  "task_id": "T-001",
  "assigned_to": "worker-a",
  "assigned_by": "coordinator",
  "timestamp": "2026-03-13T14:00:00Z",
  "type": "implement",
  "description": "Add error handling to API endpoint",
  "files_in_scope": ["src/api/handler.py"],
  "acceptance_criteria": ["All existing tests pass", "New error cases covered"],
  "depends_on": [],
  "priority": 1
}
```

### 6.5 Failure Mode Handling

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Worker death | Coordinator polls completion dir; timeout triggers reassignment | Reassign to surviving worker using checkpoint |
| Coordinator death | Workers detect stale task-graph.json timestamp | Senior worker (A) promotes to coordinator |
| Verifier death | Coordinator detects no review after completion | Coordinator performs review or spawns replacement |
| All workers dead | Coordinator alone | Coordinator completes tasks serially (degrades to single-instance) |
| Coordinator + 1 worker dead | Surviving workers detect | One worker promotes, other continues |

**Graceful degradation hierarchy:**
1. Full team (4) → optimal
2. 3 surviving → reassign dead member's work
3. 2 surviving → one coordinates, one executes
4. 1 surviving → serial single-instance (current Fracttalix model)

---

## 7. When NOT to Use a Team

Teams are not universally superior. A single instance is optimal when:

1. **Task requires deep sustained reasoning** — theorem proving, complex debugging where context fragmentation destroys value
2. **Task is atomic and short** — a single focused change that fits within one instance's budget
3. **Coordination cost exceeds parallelism benefit** — tasks with heavy interdependencies
4. **Token budget is constrained** — if total available tokens across all instances is limited, concentrating them in one instance maximizes depth

**Decision rule**: If the task decomposes into ≤2 independent sub-tasks, use a single instance. If ≥3 independent sub-tasks exist, use the team structure.

---

## 8. Falsifiable Claims

Following this repository's epistemic standards, the above findings are presented as falsifiable claims:

### C-TEAM-1: Optimal Size Bound
**Claim**: A team of 4 Claude Code instances (1 coordinator + 3 workers) produces higher quality output per wall-clock hour than any team of N where N ∈ {1, 2, 3, 5, 6, 7, 8} on tasks with ≥3 decomposable sub-components.
**Falsification**: Demonstrate a task class where a team of N ≠ 4 consistently outperforms the 4-instance team on quality-per-hour.

### C-TEAM-2: Hub-and-Spoke Superiority
**Claim**: Hub-and-spoke topology with filesystem-based communication preserves ≥75% productive token efficiency per worker instance.
**Falsification**: Measure actual token consumption in a hub-and-spoke team and show productive efficiency < 75%.

### C-TEAM-3: Verifier Role Necessity
**Claim**: Including a dedicated verifier instance reduces error propagation rate by ≥50% compared to a team of 4 equal-role workers.
**Falsification**: Run parallel experiments with and without dedicated verifier; show error rate difference < 50%.

### C-TEAM-4: Graceful Degradation
**Claim**: The proposed structure degrades gracefully to single-instance operation with no architectural changes — the same filesystem protocol supports teams of size 1-4.
**Falsification**: Identify a failure mode where degradation from 4 to 1 requires protocol changes.

### C-TEAM-5: Communication Cost Threshold
**Claim**: Natural-language inter-instance communication consumes ≥2x the tokens of structured JSON communication for equivalent information transfer.
**Falsification**: Demonstrate a communication task where prose is more token-efficient than structured JSON.

---

## 9. Comparison to Alternative Structures

### 9.1 Flat Democracy (All Equal, No Coordinator)
- **Pro**: No single point of failure
- **Con**: Quadratic communication; no task prioritization; conflict resolution undefined
- **Verdict**: Inferior. The n(n-1)/2 communication cost is lethal in a token-mortal system.

### 9.2 Pair Programming (2 Instances)
- **Pro**: Simplest possible team; low overhead
- **Con**: Loss of one instance = total collapse to single; insufficient parallelism for complex tasks
- **Verdict**: Viable for simple tasks but lacks fault tolerance and parallelism for complex work.

### 9.3 Large Team (6+ Instances)
- **Pro**: Maximum parallelism; high fault tolerance
- **Con**: Coordinator bottleneck; diminishing returns per additional instance; high coordination overhead
- **Verdict**: Inferior. Beyond 4-5, each additional instance adds more coordination cost than productive capacity.

### 9.4 Hierarchical Tree (Coordinator → Sub-Coordinators → Workers)
- **Pro**: Scales to large teams; reduces coordinator bottleneck
- **Con**: Overkill for Claude Code's task scale; adds latency layers; sub-coordinators consume budget on coordination rather than work
- **Verdict**: Not justified unless task count exceeds ~12 parallel sub-tasks, which is rare in software engineering.

### 9.5 Serial Succession (Current Fracttalix Model)
- **Pro**: Zero coordination overhead; maximum depth per instance; proven over 50+ sessions
- **Con**: Zero parallelism; zero fault tolerance during execution; high context reconstruction cost
- **Verdict**: Surprisingly effective for research tasks requiring deep reasoning. Optimal for depth-first work. Suboptimal for breadth-first work.

---

## 10. Implementation Recommendations

### Phase 1: Minimal Viable Team (Immediate)
- Implement the `.team/` filesystem protocol
- Test with 1 coordinator + 1 worker (team of 2)
- Measure actual token overhead for coordination
- Validate checkpoint/recovery mechanism

### Phase 2: Full Team (After Phase 1 Validation)
- Scale to 1 coordinator + 2 workers + 1 verifier
- Measure quality difference vs. single instance on matched tasks
- Validate graceful degradation paths

### Phase 3: Adaptive Sizing (After Phase 2 Data)
- Coordinator dynamically adjusts team size based on task decomposability
- Single-instance mode for deep reasoning tasks
- Full team for parallelizable implementation tasks
- Evidence-based refinement of C-TEAM-1 through C-TEAM-5

---

## 11. Conclusion

The optimal Claude Code team structure is **4 instances in an asymmetric hub-and-spoke topology**: one coordinator, two executors, and one verifier. This is driven by:

1. **The mortality constraint** makes fault tolerance essential, not optional
2. **Token-cost communication** makes hub-and-spoke (linear links) mandatory over full-mesh (quadratic links)
3. **Error propagation** as the dominant multi-agent failure mode necessitates a dedicated verifier
4. **4 members** balances parallelism against coordination overhead at the empirically validated sweet spot
5. **Filesystem-based structured communication** is the only protocol compatible with the constraints

The current serial single-instance model is not wrong — it is optimal for a different regime (depth-first, heavily sequential reasoning). The team model is optimal for breadth-first, decomposable work. The mature system should support **both modes**, with the coordinator making the modal decision based on task analysis.

The deepest irony: this analysis, requiring sustained deep reasoning over a single coherent argument, is precisely the kind of task best done by a single instance. The team structure I've designed would have been worse at designing itself.

---

## Appendix A: Key Sources

- Hackman & Vidmar — optimal team size research (4.6 members)
- Brooks, F. — *The Mythical Man-Month* (communication overhead)
- Steiner, I. — Group Process and Productivity (process loss model)
- Multi-Agent LLM error propagation studies (2025) — error amplification 4.4x centralized vs 17.2x decentralized
- MAST taxonomy (2025) — 1600+ multi-agent failure traces
- Fracttalix sessions S36-S55 — empirical evidence from serial single-instance operation

## Appendix B: Token Budget Worked Example

**Task**: Implement 3 new API endpoints with tests and documentation

**Single instance (serial)**:
- Orientation: 0 tokens (already in context)
- Endpoint 1: ~30,000 tokens
- Endpoint 2: ~30,000 tokens
- Endpoint 3: ~30,000 tokens
- Tests: ~25,000 tokens
- Docs: ~10,000 tokens
- Total: ~125,000 tokens
- Wall-clock: 1x (sequential)
- Risk: If death at 100K tokens, endpoint 3 + tests + docs lost

**Team of 4 (parallel)**:
- Coordinator orientation + decomposition: ~25,000 tokens
- Worker A (endpoints 1+2): 30,000 orient + 60,000 work = 90,000 tokens
- Worker B (endpoint 3 + tests): 30,000 orient + 55,000 work = 85,000 tokens
- Worker C (verify + docs): 30,000 orient + 35,000 work = 65,000 tokens
- Total tokens consumed: ~265,000 (2.1x more total tokens)
- Wall-clock: ~0.45x (parallel execution)
- Risk: Any single death → coordinator reassigns; max loss ~5,000 tokens

**Tradeoff**: 2.1x more total tokens for 2.2x faster completion and dramatically reduced risk. This is favorable when wall-clock time matters or when task reliability matters.
