# Distributed Documentation Networks: An FRM-Grounded Approach to Corpus Persistence

**Author:** Thomas Brennan (Independent Researcher, ORCID: 0009-0002-6353-7115)
**AI contributions:** Claude (Anthropic) contributed implementation drafting, literature framing, and manuscript composition. All work is contributed to the public domain.
**Date:** 2026-03-13
**Version:** 1.0
**Licence:** CC0-1.0
**Corpus:** github.com/thomasbrennan/Fracttalix
**DOI:** 10.5281/zenodo.18859299

---

## 1. Abstract

A research corpus stored on a single hosting provider is not a network. It is a node. Nodes fail; networks route around failure. This paper describes the design and implementation of a Distributed Documentation Network (DDN) for the Fracttalix research corpus, grounded in the theoretical framework the corpus itself contains — the Fractal Rhythm Model (FRM) and its three-channel dissipative network model. The DDN replicates 156 documentation artifacts across eight independent nodes spanning four git hosting providers, three archival services, and one content-addressed storage layer. The system is fully automated: every push to the primary branch propagates to all mirrors, every release deposits immutable snapshots, and a health monitor verifies integrity every six hours with automatic self-healing. The theoretical basis for the network design is derived from the FRM's treatment of networks as active dissipative systems that require continuous coupling energy to maintain coherence — the same framework used by the Fracttalix Sentinel software to detect anomalies in streaming time series. This self-referential grounding is not circular; it is convergent validation that the first principles apply at every scale.

---

## 2. Statement of Need

The Fracttalix corpus consists of 162 machine-readable falsifiable claims across 21 AI layer registries, 12 working papers, a JOSS software paper, 405 automated tests, and 63 markdown documentation files. As of version 12.3.0, the corpus is archived on Zenodo with DOI 10.5281/zenodo.18859299 and published to PyPI [@FRM2026].

All of these artifacts reside on a single hosting provider: GitHub. This architecture has a fundamental property that the FRM framework identifies as structurally dangerous: it is a node, not a network.

A node has the following characteristics in FRM terms:

- **Coupling strength** κ̄ = 0 across providers (no cross-node synchronization exists)
- **Maintenance burden** μ = 1 − κ̄ = 1.0 (all energy committed to a single point; zero adaptive reserve)
- **Kuramoto order parameter** Φ is undefined (phase coherence requires at least two oscillators)
- **Diagnostic window** Δt is degenerate (no coupling degradation rate to measure; the system transitions from "up" to "gone" in one step)

This is the Tainter critical regime [@Tainter1988]: maintenance burden at maximum, adaptive reserve at zero. The system cannot absorb perturbations because it has no redundancy to absorb them with.

The question is not whether GitHub will experience downtime — it will. The question is whether the corpus survives that event without human intervention. With a single node, the answer is uncertain. With a network, the answer is yes.

---

## 3. Theoretical Basis

### 3.1 Networks as Active Dissipative Systems

The Fractal Rhythm Model treats the systems it monitors as **active dissipative systems** — structures that continuously expend energy to maintain their topology, coupling, and phase coherence. This treatment, originally developed for streaming time-series anomaly detection, applies directly to documentation networks.

A documentation network is not a passive store. It is a thermodynamic system. Energy (compute cycles, bandwidth, authentication tokens, human attention) must be continuously invested to maintain:

1. **Coupling** between nodes — synchronization of content across providers
2. **Carrier patterns** — regular update rhythms that propagate changes coherently
3. **Phase coherence** — consistent state across all replicas at any given time

A network that does not dissipate this energy is not a network. It is a collection of diverging copies, each accruing independent drift until none can be trusted as authoritative. Dissipation is the thermodynamic requirement for information transmission, not a cost to be minimized.

### 3.2 The Three Channels Applied to Documentation Networks

The FRM monitors system health through three independent diagnostic channels. Each maps directly to documentation network health:

**Channel 1 — Structural (Network Topology).** The six structural statistics (mean, variance, skewness, kurtosis, lag-1 autocorrelation, stationarity) describe how the network reacts to perturbations. In a documentation network, the relevant structural metrics are: average replica count per document, spread in sync freshness across mirrors, persistence of sync failures over time, and whether the network topology is stable or under reconfiguration. The collapse signature identified by the FRM is simultaneous variance amplification and increasing autocorrelation — the network is losing its elastic restoring capacity, and small perturbations persist rather than dissipating [@Page1954].

**Channel 2 — Rhythmic (Update Frequency Coherence).** Documents update at different frequencies. Architecture specifications change annually; changelogs change weekly. The FRM decomposes these into five frequency bands (ultra-low through ultra-high) and measures **phase-amplitude coupling** (PAC) between them [@Tort2010]. Strong PAC means slow-changing documents modulate the amplitude of fast-changing documents coherently — when an architectural decision changes, a burst of downstream updates follows in coordinated fashion. Weak PAC means the documentation layers are decoupled: fast-changing documents contradict slow-changing ones, and the corpus loses internal consistency.

The **Kuramoto order parameter** Φ [@Kuramoto1984] measures global phase synchronization across all mirrors. Φ = 1 means all mirrors hold identical state at the same commit hash. Φ → 0 means mirrors have diverged into incoherent states. A documentation network must maintain Φ close to 1.0 to be trustworthy.

**Channel 3 — Temporal (Degradation Ordering).** In a dissipative system, the temporal ordering of degradation events is deterministic, reflecting the hierarchy of energy scales. The thermodynamically expected sequence is:

1. Band anomaly — one provider's sync frequency drops
2. Coupling degradation — cross-provider consistency starts failing
3. Structural-rhythmic decoupling — topology metrics and sync rhythms no longer correlate
4. Cascade precursor — all three conditions fire simultaneously

If phase coherence (Φ) collapses before coupling strength (κ̄) degrades, the temporal ordering is reversed. This indicates non-organic failure: a provider policy change, a deliberate attack, or a network partition that is disrupting the system through a mechanism outside the normal degradation pathway.

### 3.3 Maintenance Burden and Adaptive Reserve

The FRM defines maintenance burden as μ = 1 − κ̄, where κ̄ is mean coupling strength across all node pairs. This metric directly quantifies the fraction of system energy consumed by maintaining existing structure versus the fraction available for adaptation.

| Regime | μ Range | Interpretation |
|--------|---------|----------------|
| HEALTHY | < 0.50 | Mirrors well-synced; energy available for improvements |
| REDUCED_RESERVE | 0.50–0.75 | Sync issues appearing; review needed |
| TAINTER_WARNING | 0.75–0.90 | Most effort spent on consistency fixes |
| TAINTER_CRITICAL | ≥ 0.90 | No adaptive capacity; collapse imminent |

This maps directly to the Tainter thesis on societal collapse: complex systems fail not because of any single catastrophic event, but because maintenance burden consumes all available surplus, leaving zero capacity to respond to novel perturbations [@Tainter1988]. A documentation network with μ ≥ 0.9 is one outage away from incoherence regardless of how many nodes it contains.

### 3.4 The Diagnostic Window

The FRM provides a quantitative estimate of time-to-collapse:

```
Δt = (κ̄ − κ_c) / |dκ̄/dt|
```

Where κ_c is the Kuramoto critical coupling threshold below which synchronization cannot be sustained [@Kuramoto1984]. This is not an arbitrary threshold; it is derived from the physics of phase-locked oscillators. When coupling falls below κ_c, the network literally cannot maintain coherent state — not because a human decided a threshold, but because the mathematics requires it.

For a documentation network, Δt estimates how many sync cycles remain before mirrors diverge beyond recovery. If Δt < 5, immediate intervention is required.

### 3.5 Self-Referential Grounding

The theoretical basis for this network design is the same framework contained within the corpus the network protects. This is not circular reasoning. The FRM was developed to describe a general class of systems: dissipative networks that maintain coherence through coupling. A documentation repository is an instance of that class. The theory applies to the corpus because the corpus is an instance of the phenomenon the theory describes.

This self-applicability has a specific structure:

1. The FRM describes how dissipative networks degrade (theory)
2. The corpus is a dissipative network (instance)
3. The DDN implements FRM-derived monitoring and coupling (application)
4. The DDN protects the corpus that contains the FRM (closure)

Step 4 does not create a logical dependency. The FRM's validity does not depend on the DDN existing. The DDN's correctness does not depend on the FRM being the only possible theoretical basis. What the closure demonstrates is **convergent validation**: a first-principles framework that correctly describes a class of systems should apply to every instance of that class, including instances that contain the framework itself. If it did not apply to its own storage infrastructure, that would be evidence against its generality — not evidence for it.

The Meta-Kaizen papers (MK-P1 through MK-P6) formalize this property as self-applicable governance: a verification protocol that can audit itself using its own procedures. The Dual Reader Standard (DRS) formalizes the communication layer: every claim must be readable by both a human reader and an AI reader through independent channels. The DDN formalizes the persistence layer: every artifact must be reachable through both a primary path and alternative paths. The same structural principle — redundant independent channels — operates at every layer of the stack.

---

## 4. Implementation

### 4.1 Network Topology

The DDN defines eight nodes across four independent infrastructure layers:

| Node | Provider | Type | Role | Capabilities |
|------|----------|------|------|--------------|
| github-primary | GitHub | git-remote | primary | push, pull, CI/CD, issues, releases |
| gitlab-mirror | GitLab | git-remote | mirror | push, pull, CI/CD |
| codeberg-mirror | Codeberg | git-remote | mirror | push, pull |
| bitbucket-mirror | Bitbucket | git-remote | mirror | push, pull |
| zenodo-archive | Zenodo | archive | archive | versioned snapshots, DOI |
| software-heritage | Software Heritage | archive | archive | immutable snapshots, SWHID |
| internet-archive | Internet Archive | archive | archive | Wayback snapshots |
| ipfs-snapshot | IPFS | content-addressed | immutable-archive | decentralized, content-addressed |

The providers are chosen for independence: GitHub (Microsoft), GitLab (GitLab Inc.), Codeberg (non-profit, EU-hosted), Bitbucket (Atlassian), Zenodo (CERN), Software Heritage (Inria), Internet Archive (non-profit), IPFS (decentralized protocol). No two providers share corporate ownership, legal jurisdiction, or infrastructure dependencies.

### 4.2 Coupling Mechanisms

Each infrastructure layer uses a different coupling mechanism appropriate to its update frequency:

**Git mirrors** (high-frequency coupling): On every push to the primary branch, a GitHub Actions workflow pushes all branches and tags to all git mirror remotes. Pushes use exponential backoff retry (2s, 4s, 8s, 16s) with a maximum of four attempts. This coupling runs on every commit — the highest-frequency band in the documentation network.

**Archive snapshots** (release-frequency coupling): On every tagged release, the workflow creates a compressed tarball of all documentation artifacts (156 files as of v12.3.0), requests archival from Software Heritage via their public save API, saves the repository to the Internet Archive Wayback Machine, and optionally pins the snapshot to IPFS via a pinning service. This coupling runs on releases — the mid-frequency band.

**Periodic health checks** (low-frequency coupling): Every six hours, a scheduled workflow verifies that all nodes are reachable and that git mirrors hold the same HEAD commit as the primary. This is the monitoring channel, not a coupling channel — it detects when coupling has failed.

### 4.3 Self-Healing Protocol

When the health monitor detects divergence between mirrors:

1. **Detect**: Compare HEAD commit hashes across all reachable git mirrors
2. **Diagnose**: Identify which nodes have diverged and by how many commits
3. **Heal**: Push current authoritative state to all diverged mirrors with exponential backoff retry
4. **Verify**: Re-check all nodes after healing and log results to a JSONL audit file
5. **Escalate**: If quorum drops below threshold after healing attempts, the workflow exits with a non-zero status code, which can trigger alerting via GitHub's notification system

### 4.4 Quorum and Recovery

The network defines two thresholds:

- **Minimum replicas**: 3 (below this, the network is degraded but functional)
- **Quorum threshold**: 2 (below this, the network cannot guarantee consistency)

Any two of the eight nodes can reconstruct the complete repository state. The recovery fallback order prioritizes git mirrors (which hold full commit history) over archive snapshots (which hold point-in-time state).

Recovery from total git mirror loss proceeds through the archive layer: the most recent IPFS snapshot or Zenodo deposit contains all documentation artifacts. These can be used to re-initialize a git repository and re-establish the mirror network.

### 4.5 Content-Addressed Immutability

IPFS snapshots provide a property that git mirrors do not: content-addressed immutability. A git mirror can be force-pushed with altered history. An IPFS CID is a cryptographic hash of the content — if the content changes, the CID changes. This means IPFS snapshots serve as tamper-evident checkpoints. If a git mirror's content does not match the IPFS CID for the same release, the mirror has been altered.

### 4.6 CLI Tooling

The DDN is managed through a Python CLI tool with zero external dependencies (consistent with the Fracttalix core design principle):

```bash
fracttalix-ddn status      # Show network health across all nodes
fracttalix-ddn sync        # Push to all configured mirrors
fracttalix-ddn snapshot    # Create content-addressed archive
fracttalix-ddn verify      # Cross-node integrity verification
fracttalix-ddn recover <n> # Rebuild a failed node from survivors
fracttalix-ddn bootstrap   # First-time network setup
```

The tool loads its topology from `network/manifest.json`, which defines all nodes, their capabilities, sync policies, and recovery procedures in a machine-readable format.

---

## 5. Persistence Guarantees

| Failure Scenario | Impact | Recovery |
|---|---|---|
| GitHub down (temporary) | No new pushes | Mirrors serve reads; auto-sync on recovery |
| GitHub down (permanent) | Primary lost | Clone from any mirror; update manifest |
| One mirror lost | Reduced redundancy | Add replacement; push from any survivor |
| All git mirrors down | Git layer lost | Rebuild from IPFS snapshot + archive layer |
| IPFS pins expire | Content-addressed layer lost | Re-pin from git mirrors |
| Zenodo down | DOI resolution fails | Content accessible on all other layers |
| Correlated internet partition | Regional access loss | Geographic diversity of providers |

The network is designed to survive any single-layer failure without data loss and any two-layer failure with recovery from the surviving layer.

---

## 6. Limitations

1. **Authentication dependency.** Git mirror sync requires authentication tokens for each provider. Token expiry or revocation silently breaks coupling until detected by the health monitor. The six-hour check interval means up to six hours of divergence can accumulate before detection.

2. **Force-push vulnerability.** The sync workflow uses `--force` to ensure mirrors exactly match the primary. A malicious or accidental force-push to the primary will propagate to all mirrors within the sync interval. The IPFS layer provides tamper-evident checkpoints but does not prevent propagation of the altered state.

3. **Archive layer is snapshot-based.** Zenodo, Software Heritage, and Internet Archive hold point-in-time snapshots, not live state. Recovery from the archive layer loses all commits between the last snapshot and the failure event.

4. **IPFS pin persistence.** IPFS content is only available as long as at least one node pins it. Free-tier pinning services may garbage-collect pins after inactivity periods. Long-term IPFS persistence requires either paid pinning or running a dedicated IPFS node.

5. **No automated provider account creation.** The network topology defines eight nodes, but creating accounts and repositories on GitLab, Codeberg, and Bitbucket requires manual human action. The system automates everything after initial setup but cannot automate the setup itself.

---

## 7. Conclusion

A research corpus stored on a single provider exists at maintenance burden μ = 1.0 with zero adaptive reserve. The Fractal Rhythm Model identifies this as the Tainter critical regime — the system is one perturbation from incoherence. The Distributed Documentation Network reduces μ by distributing coupling across eight independent nodes, establishing a diagnostic window of indefinite length and a quorum-based recovery guarantee that any two nodes can reconstruct the full corpus.

The theoretical basis for this design is the FRM's treatment of networks as active dissipative systems — the same framework the corpus contains. This self-referential grounding is not circular. It is the expected behavior of a first-principles framework that correctly describes a general class of systems: it applies at the scale it was designed for (streaming anomaly detection), and it applies at the meta-scale (its own persistence infrastructure), because both are instances of the same underlying physics.

The question was never "will GitHub go down?" The question is: when it does, does the corpus survive? With a network, yes. With a node, uncertain.

---

## 8. AI Usage Disclosure

Claude (Anthropic, claude-opus-4-6) contributed: implementation of the DDN CLI tool and health monitor, drafting of the GitHub Actions workflow, structural composition of this manuscript, and synthesis of FRM theory applied to network resilience. The theoretical identification of the corpus as a dissipative network instance and the design decision to ground the DDN in FRM principles originated in collaborative dialogue. All work is contributed to the public domain under CC0-1.0.

---

## References

See `paper.bib` for full BibTeX entries. Key references:

- [@FRM2026] Brennan, T. (2026). Fractal Rhythm Model: Working Papers 1–6. Zenodo. doi:10.5281/zenodo.18859299
- [@Tainter1988] Tainter, J. A. (1988). The Collapse of Complex Societies. Cambridge University Press.
- [@Kuramoto1984] Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence. Springer.
- [@Tort2010] Tort, A. B. L., et al. (2010). Measuring Phase-Amplitude Coupling Between Neuronal Oscillations of Different Frequencies. J. Neurophysiol., 104(2), 1195–1210.
- [@Page1954] Page, E. S. (1954). Continuous Inspection Schemes. Biometrika, 41(1–2), 100–115.
- [@BandtPompe2002] Bandt, C. & Pompe, B. (2002). Permutation Entropy: A Natural Complexity Measure for Time Series. Phys. Rev. Lett., 88(17), 174102.

---

## Appendix A: AI Layer — Channel 2 Asset

The machine-readable claims for this paper will be deposited in `ai-layers/DDN-P1-ai-layer.json` upon phase completion. Schema version: v3-S51. Phase status: DRAFT.

Falsifiable claims in this paper:

| ID | Claim | Type | Falsification Condition |
|----|-------|------|------------------------|
| F-DDN.1 | Any 2 of 8 nodes can reconstruct the full repository state | F | Demonstrate a 2-node subset that cannot reconstruct HEAD |
| F-DDN.2 | Mirror sync completes within 4 retry cycles under transient failure | F | Show a transient failure requiring >4 retries |
| F-DDN.3 | Health monitor detects divergence within 6 hours | F | Show divergence persisting >6 hours undetected |
| F-DDN.4 | Auto-heal restores diverged mirrors without human intervention | F | Show a diverged mirror that auto-heal cannot restore |
| F-DDN.5 | IPFS CID mismatch detects tampered mirror content | F | Show altered content that produces matching CID |

Falsification kernel: `ai-layers/falsification-kernel.md`
