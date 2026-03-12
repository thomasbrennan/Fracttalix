# ========================================================================
# CORPUSARCH PROTOCOL AMENDMENT LOG
# Fracttalix Programme  |  Thomas Brennan
# Entwood Hollow Research Station, Trinity County CA
# Amendments S48-A1 and S48-A2  |  Session 49 formalisation  |  12 March 2026

## AMENDMENT S48-A1 — principle_10_audit Array Field

Scope:              AI Layer Schema v2 — all papers P2 and later
Session:            S48
Status:             ACCEPTED by Thomas Brennan
PR-5 formalisation: S49

Amendment ID:       S48-A1
Field added:        principle_10_audit (array)
Schema version:     v2-S42+S48 — required in all P2 and later AI layers

Trigger:
HR-1.2 (P2 Phase 1) — Principle 10 audit trail requires machine-readable
per-constant records, not just a boolean flag.

Field definition:
Array of objects. Each object: { constant_or_condition, status,
derivation_path, gap_id }. Required for every non-trivial constant or
derived condition in the paper.

Allowed status values:
LIVE_EDGE | CLOSED | OPEN_C4_TRACKING

principle_10_compliant:
Boolean. Set to true at CBT Phase 5 only when ALL principle_10_audit
entries have status LIVE_EDGE or CLOSED. No entries may remain
OPEN_C4_TRACKING at CBT close.

Circularity check:
All derivation_path entries must terminate at P1 live edges or IR-rule
axioms. Paths that loop back to the paper's own claims, or terminate at
unanchored intermediate claims, produce principle_10_compliant = false.

Backward compatibility:
P1 AI layer (law_A type) is exempt. principle_10_audit is not required in
P0 or P1. Required in all papers P2 onwards.

First applied:
P2 AI layer v1 (S48). P3 AI layer v1 (S48).

## ========================================================================
## AMENDMENT S48-A2 — I-9 Step 4 Circularity Detection

Scope:              CBT I-9 Protocol — Step 4 (Principle 10 Audit)
Session:            S48
Status:             ACCEPTED by Thomas Brennan
PR-5 formalisation: S49

Amendment ID:       S48-A2
Protocol step:      I-9 Step 4 — Principle 10 Audit

Prior Step 4:
Checked that principle_10_audit array is present and all entries have
non-null derivation_path.

Amended Step 4:
(1) Check principle_10_audit array present.
(2) For each entry: verify derivation_path is non-null.
(3) CIRCULARITY CHECK: trace each derivation_path — it must terminate at:
(a) a published live edge from a prior paper in the corpus, or
(b) an IR-rule axiom (IR-1 through IR-13).
Paths that terminate at the paper's own claim IDs, or at claim IDs
not yet live-edged, constitute a circularity violation.
(4) If any circularity detected: Step 4 FAIL —
principle_10_compliant must remain false.

Trigger:
HR-5.2 (P2 Phase 5) — circularity in principle_10_audit derivation paths
would silently pass the old Step 4 check. The amendment makes circularity
detection explicit and mandatory.

Effect on P2 CBT:
Step 4 re-executed under amended protocol. All 5 P2 principle_10_audit
entries verified non-circular. principle_10_compliant = true confirmed
under amended Step 4. No change to P2 PHASE-READY verdict.

Effect on P3 CBT:
Step 4 executed under amended protocol. All 5 P3 principle_10_audit
entries verified non-circular. principle_10_compliant = true.

First applied:
P2 Phase 5 CBT (S48). All subsequent CBT executions use amended Step 4.

## ========================================================================
## SESSION S49 GOVERNANCE EVENTS

The following governance decisions were made in Session S49 and are
registered for CorpusArch reference.

ID:          D-DIST-1
Type:        Distribution Decision
Decision:    arXiv removed from critical path. Not pursued as goal. No endorser outreach
will be conducted for platform access. Rationale: two live DOIs established;
endorsement system requires supplication inconsistent with programme posture.
Binding per Thomas Brennan S49.

ID:          D-DIST-2
Type:        Distribution Decision
Decision:    Distribution network operates on FRM network principle — no single node
indispensable. Motive: propagation and accuracy of ideas, not reputation or
institutional standing. Future platform rejections routed around without loss
of function. Binding per Thomas Brennan S49.

ID:          P11
Type:        Governance Principle
Decision:    Programme Philosophy: sole purpose is propagation and accuracy of ideas.
Hybrid human-AI team. AI instances execute, build, challenge, verify.
Thomas Brennan is sole author and decision authority. All binding decisions
require Thomas Brennan explicit acceptance. Programme will evolve with AI
capabilities. Corpus is testimony to both ideas and method.

ID:          P12
Type:        Governance Principle
Decision:    Fracttalix Network — four primary nodes:
Node 1: Claude Code — software and GitHub repository administration.
Node 2: FRM Paper Series P1-P13 — primary scholarship.
Node 3: DRS Programme — co-equal standalone programme, open scope.
Node 4: Thomas Brennan — executive function, all binding decisions.
Node 5 (corpus) is an open semantic question flagged for P9 and P13.
All nodes subject to FRM network laws. kappa = 0.91.

ID:          DN-P9-S49-1
Type:        P9 Design Note
Decision:    Distribution network, corpus dependency graph, and CBT gate structure are
structurally isomorphic. Three-layer model (content / governance / distribution)
to be formalised in P9 as governance isomorphism theorem extension.
Distribution layer is part of the corpus, not external to it.

## ========================================================================
## PR-5 COMPLETION RECORD

PR-5 Status:  COMPLETE
Executed:     Session S49
Executor:     Claude (Anthropic)

S48-A1 and S48-A2 formally logged. S49 governance events registered.
This document supersedes the informal PR-5 note in the S48 handoff.

# ========================================================================
# END  |  CorpusArch Protocol Amendment Log  |  S49
