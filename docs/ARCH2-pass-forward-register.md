# ARCH-2 Pass-Forward Register — Repo Extract

**Source:** DRP_Series_ARCH_S50.docx (canonical prose, not in repo)
**Extracted:** Session S54, 2026-03-12
**Purpose:** Machine-readable repo-canonical version of ARCH-2 pass-forward register.
Tracks all Type A axiom edges between DRP papers plus cross-programme edges.
**Authority:** Thomas Brennan. Updates require session reference.

---

## Pass-Forward Edges

### EDGE 1 — DRP-1 → DRP-2: Upstream Measurement Principle

| Field | Value |
|-------|-------|
| FROM paper | DRP-1 |
| FROM claim | C-DRP.8 |
| FROM type | D (theorem) |
| Axiom label | Upstream Measurement Principle (UMP) |
| TO paper | DRP-2 |
| RECEIVED AS | C-DRP2.6 |
| RECEIVED type | A |
| STATUS | LIVE |
| Proved in | DRP-1 v1.1 §10.7 (by contradiction) |
| AI layer sources | DRP-1 AI Layer v1.1, DRP-2 AI Layer v1.2 |
| Edge ID | EDGE-DRP2-IN-1 |

**R-13 correction (S53/S54):** Prior ARCH-2 entries citing "C-DRP.7 = UMP" were
incorrect. C-DRP.7 is the CORPUS-COMPLETE Definition. UMP is C-DRP.8.
Corrected in this extract per R-13 resolution.

---

### EDGE 2 — DRP-1 → DRP-2: Falsification Kernel (structural vocabulary)

| Field | Value |
|-------|-------|
| FROM paper | DRP-1 |
| FROM claim | C-DRP.5 |
| FROM type | F |
| Axiom label | Falsification Kernel 4-tuple (P, O, M, B) |
| TO paper | DRP-2 |
| RECEIVED AS | (structural vocabulary — not registered as named claim) |
| STATUS | LIVE |
| AI layer sources | DRP-1 AI Layer v1.1, DRP-2 AI Layer v1.2 |
| Edge ID | EDGE-DRP2-IN-2 |

---

### EDGE 3 — DRP-2 → DRP-3: Testability Relation

| Field | Value |
|-------|-------|
| FROM paper | DRP-2 |
| FROM claim | C-DRP2.4 |
| FROM type | D (definition) |
| Axiom label | Testability Relation T(P, O, M) |
| TO paper | DRP-3 |
| RECEIVED AS | C-DRP3.1 |
| RECEIVED type | A |
| STATUS | LIVE |
| AI layer sources | DRP-2 AI Layer v1.2, DRP3 D3-1.0 claim identification |
| Edge ID | EDGE-DRP2-OUT-1 |

---

### EDGE 4 — DRP-2 → DRP-4: Binary Epistemic Conservation Law

| Field | Value |
|-------|-------|
| FROM paper | DRP-2 |
| FROM claim | C-DRP2.2 |
| FROM type | F |
| Axiom label | Binary epistemic conservation law — I(M(O);P)=0 for binary M when O=f(P) |
| TO paper | DRP-4 |
| RECEIVED AS | (not yet assigned — DRP-4 not deployed) |
| RECEIVED type | A (expected) |
| STATUS | LIVE |
| Scope note | Binary-scoped in DRP-2; DRP-4 generalises to all M |
| AI layer sources | DRP-2 AI Layer v1.2 |
| Edge ID | EDGE-DRP2-OUT-2 |

---

### EDGE 5 — P2 → DRP-3: Hopf Stability Criterion (cross-programme, via ARCH-3)

| Field | Value |
|-------|-------|
| FROM paper | P2 (FRM corpus) |
| FROM claim | C-2.1 |
| FROM type | F |
| Axiom label | Hopf stability criterion — β=1/2 derivation, μ<0 condition |
| TO paper | DRP-3 |
| RECEIVED AS | C-DRP3.2 |
| RECEIVED type | A |
| STATUS | LIVE |
| Cross-programme | FRM → DRS (managed via ARCH-3 cross-reference map) |
| AI layer sources | P2 AI Layer, DRP3 D3-1.0 claim identification |

---

### EDGE 6 — DRP-3 → DRP-4: Convergence Principle (PLACEHOLDER)

| Field | Value |
|-------|-------|
| FROM paper | DRP-3 |
| FROM claim | C-DRP3.5 |
| FROM type | F (central claim) |
| Axiom label | M↔C2 Isomorphism / Convergence Principle |
| TO paper | DRP-4 |
| RECEIVED AS | (PLACEHOLDER — not yet assigned) |
| RECEIVED type | A (expected) |
| STATUS | PLACEHOLDER |
| Resolution condition | Promoted to LIVE when DRP-3 reaches PHASE-READY |
| AI layer sources | DRP3 D3-1.0 claim identification |

---

## Summary

| # | FROM | FROM CLAIM | TO | RECEIVED AS | STATUS |
|---|------|------------|----|-------------|--------|
| 1 | DRP-1 | C-DRP.8 (UMP) | DRP-2 | C-DRP2.6 | LIVE |
| 2 | DRP-1 | C-DRP.5 (Kernel) | DRP-2 | (structural) | LIVE |
| 3 | DRP-2 | C-DRP2.4 (Testability) | DRP-3 | C-DRP3.1 | LIVE |
| 4 | DRP-2 | C-DRP2.2 (Conservation) | DRP-4 | (unassigned) | LIVE |
| 5 | P2 | C-2.1 (Hopf) | DRP-3 | C-DRP3.2 | LIVE |
| 6 | DRP-3 | C-DRP3.5 (M↔C2) | DRP-4 | (unassigned) | PLACEHOLDER |

**Total edges:** 6 (5 LIVE, 1 PLACEHOLDER)
**C-DRP.7 occurrences in this file:** 0 (R-13 correction applied)

---

## Version History

| Version | Session | Changes |
|---------|---------|---------|
| v1 (extract) | S54 | Initial repo extract from .docx ARCH-2 register + live AI layer dependency edges. C-DRP.7→C-DRP.8 correction applied (R-13). Created per WO-S54-ARCH2-1. |

---

*ARCH-2 pass-forward register extracted to repo by Claude Code, Session S54.*
*Canonical prose source: DRP_Series_ARCH_S50.docx. This extract supersedes*
*the .docx for all claim ID references tracked in-repo.*
