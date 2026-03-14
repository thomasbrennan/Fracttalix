# ========================================================================
# AI LAYER  |  P1  |  The Fractal Rhythm Model: Theoretical Foundations

## DOCUMENT METADATA

Paper ID:          P1
Series Position:   Paper 1 of 13
Paper Type:        law_A
Version:           v12
Produced Session:  S49
Produced By:       Claude (Anthropic) — PR3-2 post-PHASE-READY update (P3 PHASE-READY resolves PH-1.2, PH-1.3)
Schema Version:    v2-S42
Schema URL:        https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json
Licence:           CC BY 4.0
Update Note:       Additive update only. PH-1.2 and PH-1.3 resolved by P3 C-3.REG (S48). PH-1.1 remains unresolved — pending PR-1 (Thomas action: C-2.4 empirical confirmation, Thomas sign-off required).

## PHASE-READY STATUS

Verdict:           PHASE-READY
C1:                SATISFIED
C2:                SATISFIED
C3:                SATISFIED
C4 Mode:           PHASE-READY-TRACKING
C5:                SATISFIED
C6:                SATISFIED
Placeholder Count: 1
Note:              One live placeholder — non-blocking. PH-1.1: beta=1/2 class-level empirical test pending P2 C-2.4 (Substrate Independence — analytic proof complete in F-1.4; empirical confirmation across substrate classes requires P2 C-2.4 via P3 C-3.REG; pending Thomas PR-1 sign-off). PH-1.2 and PH-1.3 resolved S48 by P3 C-3.REG (additive update PR3-2). CBT I-9 all 7 steps passed — Session 44.

## CLAIM REGISTRY

Claim ID:   A-1.1
Type:       A  [TYPE:A]
Name:       Thermodynamic irreversibility
Statement:  Systems evolve toward higher entropy; time has a preferred direction.

Claim ID:   A-1.2
Type:       A  [TYPE:A]
Name:       Information distinguishability
Statement:  Distinguishable states carry information; indistinguishable states do not.

Claim ID:   A-1.3
Type:       A  [TYPE:A]
Name:       Network definition
Statement:  A network is a set of nodes and directed edges; coupling κ is the ratio of active to possible edges.

Claim ID:   A-1.4
Type:       A  [TYPE:A]
Name:       Non-equilibrium physics
Statement:  Systems driven from equilibrium exhibit emergent structure at phase transitions.

Claim ID:   A-1.5
Type:       A  [TYPE:A]
Name:       Substrate independence
Statement:  A result derived from topology and information content without reference to physical substrate applies to all substrates satisfying those conditions.

Claim ID:   D-1.1
Type:       D  [TYPE:D]
Name:       FRM functional form
Statement:  f(t) = B + A·e^(−λt)·cos(ωt+φ), where B is baseline, A is initial amplitude, λ is decay rate, ω is characteristic frequency, φ is phase offset.

Claim ID:   D-1.2
Type:       D  [TYPE:D]
Name:       Characteristic frequency
Statement:  ω = π/(2·τ_gen), where τ_gen is the substrate-specific generation timescale. Derived from Hopf quarter-wave theorem (Session 43). β = 1/2 is not fitted.

Claim ID:   D-1.3
Type:       D  [TYPE:D]
Name:       Decay rate
Statement:  λ ≈ |α| / (Γ·τ_gen), where α is the normalized distance from the Hopf bifurcation and Γ = 1+π²/4 ≈ 3.467 is the universal loop impedance constant. Leading-order perturbation expansion. Mean error 3.61%.

Claim ID:   D-1.4
Type:       D  [TYPE:D]
Name:       Validation set P1
Statement:  Confirmed substrates (n=3): mammalian circadian (tau_gen=6 hr), cyanobacterial circadian (tau_gen~5.5 hr), Ibn Khaldun dynastic cycle (tau_gen=27.5 yr). Candidate substrates (n=11) tabulated in Appendix A (D-1 S47). Confirmed timescale span: 4.6 OOM (cyanobacteria to dynastic cycle). Confirmed+candidate span: 11.1 OOM (neural gamma to Roman republican cycle). Formal validation of candidate substrates scoped to Papers 3-7.

Claim ID:   F-1.1
Type:       F  [TYPE:F]
Name:       FRM functional form uniqueness
Statement:  The FRM functional form f(t)=B+A·e^(−λt)·cos(ωt+φ) is the unique solution for network information transmission across time satisfying conditions C1–C4.

Claim ID:   F-1.2
Type:       F  [TYPE:F]
Name:       OOM span validation
Statement:  FRM validated across confirmed substrates spanning 4.6 OOM in characteristic timescale (cyanobacterial circadian to Ibn Khaldun dynastic cycle). Candidate substrates extend span to 11.1 OOM; formal validation scoped to Papers 3-7.

Claim ID:   F-1.3
Type:       F  [TYPE:F]
Name:       β = 1/2 substrate independence (empirical)
Statement:  The critical exponent β = 1/2 is substrate-independent across all five domain classes.

Claim ID:   F-1.4
Type:       F  [TYPE:F]
Name:       β = 1/2 analytic derivation (Hopf quarter-wave theorem)
Statement:  β = 1/2 follows necessarily from the Hopf criticality condition of a normalized first-order DDE with delayed negative feedback. This is a theorem, not an empirical result.

Claim ID:   F-1.5
Type:       F  [TYPE:F]
Name:       λ derivation — leading order
Statement:  The FRM damping rate λ ≈ |α|/(Γ·τ_gen) where Γ = 1+π²/4 ≈ 3.467 is the universal loop impedance constant, derived from the DDE characteristic equation via perturbation expansion about the Hopf critical point.

Claim ID:   F-1.6
Type:       F  [TYPE:F]
Name:       Circadian period prediction
Statement:  T = 4·τ_gen predicts the mammalian circadian period T = 24 hr from τ_gen = 6 hr with no fitted parameters. τ_gen independently measured by four molecular biology sources.

Claim ID:   F-1.7
Type:       F  [TYPE:F]
Name:       Stuart-Landau connection
Statement:  FRM is the approximate transient solution of the Stuart-Landau normal form in the linear regime for μ < 0, with λ ≈ |μ|. Nonlinear terms introduce additional damping (slope k=1.10). The FRM scope boundary coincides with the Hopf bifurcation.

## PLACEHOLDER REGISTER

ID:              PH-1.1
Source Claim:    F-1.3
Description:     beta=1/2 class-level empirical test. Pending P2 C-2.4 (Substrate Independence — empirical confirmation across substrate classes). Note: the analytic proof (F-1.4) is complete — PH-1.1 concerns only the class-level 2-sigma empirical confirmation across all five domain classes. C-2.4 resolves this via P3 C-3.REG protocol.
Resolved:        False
Blocks Phase:    False
Target Paper:    P2
Target Claim:    C-2.4
Target Session:  OPEN — resolves with P2 completion

ID:              PH-1.2
Source Claim:    F-1.1
Description:     F-1.1 WHERE clause references 'P3 Section 4 regression' (R2_frm, R2_best_alt). Pending P3 measurement and diagnostics paper (Phase 1). Until P3 is built, the regression protocol is specified by intention but not by a live claim ID.
Resolved:        True
Blocks Phase:    False
Target Paper:    P3
Target Claim:    C-3.REG
Target Session:  RESOLVED S48
Resolution:      RESOLVED S48 — P3 PHASE-READY. C-3.REG (FRM Measurement and Diagnostics) provides the regression protocol referenced in F-1.1 WHERE clause. P3-ai-layer-v2.json published to GitHub.
Res. Session:    S48

ID:              PH-1.3
Source Claim:    F-1.1
Description:     F-1.2 WHERE clause references 'P3 Section 4 protocol' (T_char, n_passing) and 'P1 Appendix B' (R2_best_alt). Both reference the P3 measurement protocol. Pending P3 Phase 1.
Resolved:        True
Blocks Phase:    False
Target Paper:    P3
Target Claim:    C-3.REG
Target Session:  RESOLVED S48
Resolution:      RESOLVED S48 — P3 PHASE-READY. C-3.REG provides the measurement protocol (T_char, n_passing) referenced in F-1.2 WHERE clause. P3-ai-layer-v2.json published to GitHub.
Res. Session:    S48

# ========================================================================
# END  |  P1  |  v12
