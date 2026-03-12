# R-13 RESOLUTION — C-DRP.8 STATEMENT CORRECTION
# DRP-2 v1.2 Prose Read Report
# Session S54 · 12 March 2026

---

## R-13 — ISSUE 1 RESOLUTION

SOURCE VERIFIED: DRP-2 v1.2 §1 (lines 10, 19) reproduces the
DRP-1 v1.1 UMP theorem statement as follows:

> "The UMP states that the independence of the observation class
> O from the proposition P under test is a necessary condition
> for predicate non-vacuity." (§Abstract)

> "DRP-1 v1.1 proved the Upstream Measurement Principle: for any
> falsification predicate K = (P, O, M, B), the causal
> independence of O from P is a necessary condition for
> non-vacuity. The proof is by contradiction." (§1)

This is the authoritative statement of the UMP as received by
DRP-2 from DRP-1. DRP-1 v1.1 §10.7 is the source; DRP-2 §1
is the faithful reproduction of that source.

### Errors in prior C-DRP.8 statement (committed S53, corrected S54):

**Error 1 — WRONG INDEPENDENCE OBJECT**
Current: "the observation O must be defined independently of the model M being tested"
Correct: O must be causally independent of P (the proposition).
The UMP is about O ⊥ P, not O ⊥ M.

**Error 2 — IMPORTED CONSERVATION LAW FORMULA**
Current: "Formally: I(M(O); P) = 0 is required for any valid predicate"
Correct: I(M(O); P) = 0 is the CONSEQUENCE of violating the UMP (C-DRP2.2,
epistemic conservation law), not the UMP itself.

**Error 3 — IMPORTED DRP-3 CONCEPT**
Current: "M-termination is not achievable"
Correct: M-termination belongs to DRP-3's derivation (M↔C2 isomorphism).

### Corrected C-DRP.8 statement (applied S54):

"For any falsification predicate K = (P, O, M, B), the causal
independence of O from P is a necessary condition for non-vacuity.
This is a theorem proved by contradiction in DRP-1 v1.1 §10.7,
not a methodological convention. A predicate where O is derived
from or conditioned on P is structurally vacuous regardless of
the contents of M or B: no measurement procedure can extract
information about P from observations defined relative to P."

Source authority: DRP-2 v1.2 §1 reproducing DRP-1 v1.1 §10.7.

---

## DRP-2 PROSE READ — ADDITIONAL OBSERVATIONS

### OBS-1 — PATCH 4 NOT YET REFLECTED IN UPLOADED FILE

Lines 93 and 104 of DRP2_v1_2_S49.docx still read C-DRP.7 (UMP).
Patch 4 from S53 specified correcting these to C-DRP.8.
ACTION: Verify Patch 4 applied to prose docx. If not, apply C-DRP.7 → C-DRP.8
in §5.4 and §7 only.

### OBS-2 — PROSE VERSION HEADER INCONSISTENCY

File header reads "DRP-2 v1.1" but canonical version is v1.2.
Appendix A reads "v1.1 additions" — should be "v1.2 additions."
ACTION: Update header and Appendix A version strings.

### OBS-3 — AI LAYER NOT IN APPENDIX A (PROSE ONLY)

Appendix A describes AI layer in prose only — no structured JSON.
Consistent with R-12 (DRP-2 AI layer not yet deposited). No additional action.

### OBS-4 — CLAIM COUNT CONFIRMATION FOR R-12

DRP-2 v1.2 prose confirms canonical claim structure:

- Type A (received): 1 — UMP from DRP-1 (now C-DRP.8)
- Type D (defined): 2 — C-DRP2.4 (Testability relation) + one unnamed
- Type F (falsifiable): 9 — C-DRP2.1 through C-DRP2.9

**Canonical claim count: 1A + 2D + 9F = 12 total claims**

Appendix A "six Type F" refers to v1.1 base only; v1.2 adds 3 more.
Summary not updated from v1.1 (consistent with OBS-2).

Registry impact when R-12 resolved: current 153 + 12 = 165 total claims.

---

## GATE STATUS

- R-13: RESOLVED (C-DRP.8 statement corrected S54)
- R-12: OPEN — DRP-2 AI layer (12 claims) still required
- DRP-3 Phase 1: HELD on R-12 only

---

*Prose read report produced S54, archived by Claude Code.*
