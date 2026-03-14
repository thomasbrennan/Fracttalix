# Distributed Docs Network: FRM-Grounded Theory

**Version**: 1.0
**Grounding**: Fractal Rhythm Model (FRM) Three-Channel Dissipative Network
**Status**: Foundational specification

---

## 1. The Problem: Node vs. Network

A single repository on a single provider is a **node**. A node has zero
fault tolerance. When the node goes down, everything it held is unreachable.

> A node structure prevents loss. No structure invites loss.

The question is not whether GitHub will experience downtime — it will. The
question is whether the documentation, AI layers, and research corpus survive
that event without human intervention.

The answer requires a **network** — multiple independent nodes with sustained
coupling between them.

---

## 2. FRM Contribution: Why Networks Are Dissipative Systems

The Fractal Rhythm Model provides a theoretical framework that treats networks
as **active dissipative systems**. This is the critical insight: a network is
not a passive structure that stores data. A network is a thermodynamic system
that continuously expends energy to maintain:

1. **Coupling** between nodes (synchronization of content)
2. **Carrier patterns** (regular update rhythms across providers)
3. **Phase coherence** (consistent state across frequency bands of activity)

Dissipation is not a failure mode. It is the thermodynamic **requirement** for
information transmission. A network that doesn't dissipate energy is a
collection of disconnected nodes.

### 2.1 The Three Channels Applied to DDN

The FRM three-channel model maps directly to documentation network health:

#### Channel 1 — Structural (Network Topology)

Monitors the topology of the network itself:

| FRM Metric | DDN Metric |
|---|---|
| Mean | Average replica count per document |
| Variance | Spread in sync freshness across mirrors |
| Skewness | Asymmetry in provider reliability |
| Kurtosis | Fat-tail risk of correlated failures |
| Lag-1 autocorrelation | Persistence of sync failures over time |
| Stationarity | Whether the network topology is stable |

**Collapse signature**: When variance amplifies and autocorrelation increases
simultaneously, the network is losing its elastic restoring capacity. Small
perturbations (one mirror timing out) persist and propagate rather than
dissipating.

#### Channel 2 — Rhythmic (Update Frequency Coherence)

Documents and artifacts update at different frequencies. The FRM five-band
decomposition maps to documentation rhythms:

| Band | Documentation Layer | Update Cadence |
|---|---|---|
| Ultra-low | Architecture specs, theory papers | Annually |
| Low | API references, JOSS paper | Quarterly |
| Mid | Tutorials, examples, AI layers | Monthly |
| High | Changelog, session notes | Weekly |
| Ultra-high | CI/CD logs, health checks | Continuously |

**Phase-Amplitude Coupling (PAC)**: Slow-changing documents (architecture)
should modulate the amplitude of fast-changing documents (changelogs). When
a major architectural change occurs, it should produce a burst of downstream
updates. Strong PAC between bands means the documentation network maintains
harmonic coherence — updates cascade in coordinated fashion.

**Kuramoto Order Parameter (Φ)**: Measures whether all mirrors are
phase-coherent. Φ = 1 means all mirrors have identical content at the same
commit hash. Φ → 0 means mirrors have diverged into incoherent states.

#### Channel 3 — Temporal (Degradation Ordering)

In a dissipative network, the temporal ordering of degradation events is
deterministic. The thermodynamically expected sequence:

1. **Band anomaly**: One provider's sync frequency drops
2. **Coupling degradation**: Cross-provider consistency starts failing
3. **Structural-rhythmic decoupling**: Topology metrics and sync rhythms
   no longer correlate
4. **Cascade precursor**: All three conditions fire simultaneously

**Reversed sequence detection**: If coherence (Φ) collapses before coupling
degrades, the failure is non-organic — likely external intervention (provider
policy change, network partition, deliberate attack).

---

## 3. Resilience Principles Derived from FRM

### Principle 1: Maintenance Burden as Resilience Metric

```
μ = 1 − κ̄
```

Where κ̄ is mean coupling strength across all mirror pairs.

| Regime | μ Range | DDN Interpretation |
|---|---|---|
| HEALTHY | < 0.50 | Mirrors well-synced, energy for improvements |
| REDUCED_RESERVE | 0.50–0.75 | Sync issues appearing, review needed |
| TAINTER_WARNING | 0.75–0.90 | Most effort spent on consistency fixes |
| TAINTER_CRITICAL | ≥ 0.90 | No adaptive capacity, collapse imminent |

A network with μ ≥ 0.9 is spending all its energy on maintenance and has
no reserve for adaptation. This is the **Tainter critical regime** — the
network is one perturbation away from collapse.

### Principle 2: Diagnostic Window for Proactive Intervention

```
Δt = (κ̄ − κ_c) / |dκ̄/dt|
```

This estimates how many sync cycles remain before the network cannot maintain
coherence. If Δt < 5, immediate intervention is required.

### Principle 3: Redundancy Requires Coupling

Redundant copies without coupling are worse than useless — they increase
maintenance burden without improving resilience. Uncoordinated mirrors diverge,
creating contradictory states that erode trust.

True resilience requires:
- Multiple independent copies **AND**
- Automated coupling between them (sync pipelines)
- Phase coherence monitoring (integrity verification)
- Self-healing when divergence is detected

### Principle 4: Multi-Signal Cascade Gate

The DDN health monitor fires a critical alert **only** when all three
independent failure channels activate simultaneously:

1. Coupling degradation (sync failures across providers)
2. Structural-rhythmic decoupling (topology and rhythm metrics diverge)
3. EWS elevation (variance amplification + critical slowing)

This prevents false alarms from transient provider outages while maintaining
high sensitivity to true cascade events.

### Principle 5: Entropy Always Increases — Control the Rate

Documentation networks inevitably drift toward incoherence (Second Law). The
goal is not to prevent entropy increase but to control its rate through:

- Regular automated sync (every push, plus periodic)
- Content-addressed snapshots (IPFS CIDs as immutable checkpoints)
- Integrity verification (hash comparison across mirrors)
- Self-healing (auto-push to diverged mirrors)

### Principle 6: Hierarchical Frequency Coupling

The five-band decomposition suggests organizing persistence strategies by
frequency:

- **Ultra-low** (theory, architecture): Archived to Zenodo, Software Heritage
  with DOI — designed for decade-scale persistence
- **Low** (API, papers): Mirrored to 4+ git providers, content-addressed on IPFS
- **Mid** (tutorials, AI layers): Synced on every push to all mirrors
- **High** (changelog, journal): Synced to git mirrors, snapshotted on release
- **Ultra-high** (CI/CD, health): Ephemeral, not archived, but monitored

---

## 4. DDN Architecture (FRM-Grounded)

```
                         ┌─────────────────┐
                         │  LOCAL REPO      │
                         │  (authoritative) │
                         └────────┬────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
              ┌─────▼─────┐ ┌────▼─────┐ ┌─────▼─────┐
              │  GitHub    │ │  GitLab  │ │  Codeberg │ ← Git mirrors
              │  (primary) │ │  (mirror)│ │  (mirror) │   (κ̄ coupling)
              └─────┬─────┘ └────┬─────┘ └─────┬─────┘
                    │            │             │
              ┌─────▼─────┐     │       ┌─────▼──────┐
              │ Bitbucket  │     │       │   IPFS     │ ← Content-addressed
              │  (mirror)  │     │       │   (CID)    │   (Φ coherence)
              └────────────┘     │       └────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
              ┌─────▼─────┐ ┌───▼──────┐ ┌──▼───────────┐
              │  Zenodo    │ │ Internet │ │  Software    │ ← Archive layer
              │  (DOI)     │ │ Archive  │ │  Heritage    │   (ultra-low band)
              └────────────┘ └──────────┘ └──────────────┘
```

### Coupling Mechanisms

| Layer | Mechanism | Frequency | Automation |
|---|---|---|---|
| Git mirrors | `git push --all --force` | Every push to main | GitHub Actions |
| IPFS snapshots | `tar + ipfs add + pin` | Every release | GitHub Actions |
| Zenodo | DOI-linked deposit | Tagged releases | Manual + webhook |
| Software Heritage | Save API request | Every release | GitHub Actions |
| Internet Archive | Wayback save | Every release | GitHub Actions |

### Self-Healing Protocol

1. **Detect**: Health monitor checks all nodes every 6 hours
2. **Diagnose**: Compare HEAD commits across all git mirrors
3. **Heal**: Push current state to any diverged mirror (exponential backoff)
4. **Verify**: Re-check after healing, log results
5. **Alert**: If quorum drops below threshold, escalate

---

## 5. Persistence Guarantees

| Failure Scenario | Impact | Recovery |
|---|---|---|
| GitHub down (temporary) | No new pushes | Mirrors serve reads; auto-sync on recovery |
| GitHub down (permanent) | Primary lost | Clone from any mirror; update manifest |
| One mirror lost | Reduced redundancy | Add replacement; push from any survivor |
| All mirrors down | Git layer lost | Rebuild from IPFS snapshot + archive layer |
| IPFS pins expire | Content-addressed layer lost | Re-pin from git mirrors |
| Zenodo down | DOI resolution fails | Content still on all other layers |
| Internet splits | Regional access loss | Geographic diversity of providers |

**Minimum survival**: Any **two** of the eight nodes can reconstruct the
complete repository state. This exceeds the quorum threshold (2) defined in
the network manifest.

---

## 6. Theoretical Conclusion

The FRM three-channel model demonstrates that resilient networks are not
merely collections of redundant nodes. They are **coupled dissipative systems**
that require continuous energy investment to maintain coherence. The DDN
implements this insight through:

1. **Coupling** (automated sync) rather than passive copying
2. **Multi-channel monitoring** (structural + rhythmic + temporal) rather than
   single-metric health checks
3. **Thermodynamic ordering** awareness — knowing *which* failures precede
   *which* tells you where you are in the degradation sequence
4. **Adaptive reserve** tracking — when μ approaches 1.0, the network is
   brittle regardless of node count

A network that respects these principles will persist indefinitely, not because
it cannot fail, but because it routes around failure automatically and maintains
sufficient adaptive reserve to absorb novel perturbations.

> The question was never "will GitHub go down?"
> The question is: "when it does, does the corpus survive?"
> With a network: yes. With a node: uncertain.
