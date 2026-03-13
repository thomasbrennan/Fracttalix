# DRP-5 Hostile Review — HR-D5

**Paper:** DRP-5 — Underdetermination and the Relational Account
**Session:** S57
**Date:** 2026-03-13
**Review scope:** Full build (Phases 1-2)
**Review questions (from build plan):**
1. Does DRP-5 resolve the DQ problem or relocate it?
2. Does the regress termination argument hold under adversarial pressure?
3. Is C-DRP5.5 (C3 as DQ solution) falsifiable independently of C-DRP5.3?
4. Core adversarial challenge: if T2 must satisfy UMP, and T2 is theory-laden with respect to T3, is the regress actually terminated — or merely deferred one level?

---

## Challenge 1: Resolution vs Relocation

**Challenge:** DRP-5 claims to give precise DQ boundary conditions. But does it resolve the DQ problem or merely relocate it? The answer "T2 must satisfy UMP with respect to P" just pushes the question one level up: instead of "is this test of P valid?", we now ask "is this test of T2's independence valid?" The DQ problem reappears at the meta-level.

**Assessment:** This challenge is **partially sustained**. DRP-5 does relocate the DQ problem — but it relocates it to a *tractable* location. The original DQ problem is stated at the level of "can we attribute a test failure to P vs auxiliary hypotheses?" — an intractable attribution problem. DRP-5 relocates it to "is T2 causally independent of P?" — a structural question about the test construction, not about interpreting a failure. The relocation is not a defect; it is the contribution.

**Resolution:** Add explicit scope statement to C-DRP5.2: "DRP-5 does not claim to dissolve the DQ problem. It reduces the DQ problem to a precise, checkable structural condition on T2. The philosophical question of whether auxiliary hypotheses can be fully eliminated is out of scope. DRP-5 provides the *boundary condition* under which the DQ problem is tractable."

**Status: RESOLVED — scope statement added to C-DRP5.2 below.**

---

## Challenge 2: Regress Termination Under Adversarial Pressure

**Challenge:** The regress argument (D5-1.2) claims the chain terminates at "direct physical measurement independent of P." But what counts as "direct physical measurement"? A thermometer reading relies on theories of thermal expansion. A spectrometer relies on theories of electromagnetic radiation. Every measurement is theory-laden (Hanson 1958, Kuhn 1962). IR-DRP5-3 (Causal Grounding) may be question-begging: it defines "grounded" as "independent of P," which is the condition to be established.

**Assessment:** This challenge is **sustained**. The regress argument must acknowledge theory-ladenness of measurement and provide a non-circular termination criterion.

**Resolution:** Amend C-DRP5.4 to include an explicit termination criterion:

The chain terminates at T_n when:
(a) T_n is a measurement protocol whose outputs are determined by physical processes that do not reference P in their causal mechanism, AND
(b) T_n's own theoretical foundations (thermodynamics, electromagnetism, etc.) have been independently validated in contexts where P is not at issue.

Condition (b) is the answer to theory-ladenness: the theories underlying the measurement instrument are not tested by this particular test of P — they are validated independently in their own domain. The thermometer's theory of thermal expansion is not at issue when we use the thermometer to measure tau_gen.

This is a *practical* termination criterion, not a foundationalist claim about theory-free observation. The regress terminates not because we reach bedrock, but because we reach a level where the theories in play have been independently validated and are not at issue in the current test.

**Status: RESOLVED — C-DRP5.4 amended below.**

---

## Challenge 3: Independent Falsifiability of C-DRP5.5

**Challenge:** Is C-DRP5.5 (C3 as DQ solution) independently falsifiable from C-DRP5.3 (T2 ⊥ P requirement)? If C-DRP5.5 is just "C3 satisfies C-DRP5.3," then falsifying C-DRP5.3 automatically falsifies C-DRP5.5. They would not be independently falsifiable.

**Assessment:** This challenge is **partially sustained**. C-DRP5.5 is logically dependent on C-DRP5.3 (it applies C-DRP5.3 to the specific case of FRM C3). However, C-DRP5.5 has independent empirical content: it claims that the specific measurement protocols in P3 D-3.2 actually achieve T2 ⊥ P. This is falsifiable by finding a P3 measurement protocol where tau_gen extraction depends on the FRM prediction.

**Resolution:** State the joint/independent falsifiability structure explicitly:
- C-DRP5.3 falsified → C-DRP5.5 falsified (logical dependency: if the condition is wrong, the instance is wrong)
- C-DRP5.5 falsified ↛ C-DRP5.3 falsified (C3 could fail the condition without the condition being wrong)
- Independent path for C-DRP5.5: find a P3 tau_gen extraction where the output depends on whether the FRM prediction is true. E.g., if spectral sub-protocol uses T_obs/4 and T_obs is selected based on FRM prediction rather than observed dominant peak, C3 fails.

**Status: RESOLVED — falsifiability structure stated below.**

---

## Challenge 4: Core Adversarial — Is the Regress Deferred, Not Terminated?

**Challenge:** Even with the amended termination criterion, consider: the structural sub-protocol in P3 D-3.2 uses "published system architecture" to determine tau_gen. But the published system architecture relies on biological theory (T3). Biological theory relies on biochemical theory (T4). This is exactly the regress. Is it actually terminated, or have we just given it a name ("Causal Grounding") and declared victory?

**Assessment:** This challenge is the strongest and is **partially sustained**. The answer must be honest: the regress is terminated *for practical purposes* by the amended criterion (Challenge 2 resolution), not by foundationalist argument. DRP-5 does not claim to solve the philosophical problem of infinite regress in scientific knowledge. It claims that the *DQ attribution problem* — which DRP paper P or which background theory T2 is responsible for a test failure — is resolved once T2 is grounded in independently validated measurement.

The key insight: the DQ problem is not "is all science ultimately justified?" It is "can we attribute this specific test failure?" DRP-5 resolves the specific problem, not the general one.

**Resolution:** Add scope limitation to Section 6 (Scope and Limitations):

"DRP-5 does not resolve the general philosophical problem of epistemic regress (Agrippa's trilemma). The regress termination argument (C-DRP5.4) is a practical criterion for the specific DQ attribution problem: given a test failure, can we determine whether P or T2 is responsible? The criterion terminates the regress at independently validated measurement protocols — not at theory-free observation (which DRP-5 does not claim to exist). The criterion is sufficient for the DQ attribution problem because the attribution question is finite and specific: it asks about a particular test, not about the foundations of all empirical knowledge."

**Status: RESOLVED — scope limitation stated.**

---

## Challenge 5: Spectral Sub-Protocol Circularity Risk

**Challenge:** P3 D-3.2 spectral sub-protocol: tau_gen = T_obs/4 where T_obs is the dominant period from the power spectrum. But T_char = 4·tau_gen IS the FRM prediction. So if tau_gen = T_obs/4, then T_char = T_obs — i.e., the measurement is tautologically equal to the prediction. Does this make the spectral sub-protocol circular?

**Assessment:** This challenge is **dismissed after analysis**. The spectral sub-protocol does not use the FRM prediction; it measures T_obs from the raw power spectrum independently. T_obs is an empirical observable. tau_gen = T_obs/4 is a definition (the quarter-wave relationship). The FRM prediction is then T_char = 4·tau_gen = T_obs. The *test* is whether this relationship holds for other substrates, not whether it holds tautologically for the system used to extract tau_gen. The P3 protocol explicitly states "tau_gen is never fitted to O(t)" — the spectral sub-protocol extracts tau_gen from the time series, then the FRM prediction is compared to independent data.

However, there is a genuine concern: if the spectral sub-protocol is the ONLY available method for a given system, and the test compares T_char to T_obs from the SAME system, the test is tautological. P3 addresses this via the sub-protocol hierarchy (structural > spectral > mechanistic) — the spectral sub-protocol is a fallback when independent structural measurement is unavailable.

**Resolution:** Note in C-DRP5.5 that the spectral sub-protocol is the weakest of the three for DQ purposes and that structural sub-protocol is preferred precisely because it provides the strongest T2 ⊥ P guarantee. This is already implicit in D-3.2's hierarchy ordering but should be made explicit in DRP-5 prose.

**Status: RESOLVED — noted for prose Section 5.**

---

## HR-D5 Summary

| # | Challenge | Verdict | Resolution |
|---|-----------|---------|------------|
| 1 | Resolution vs relocation | Partially sustained | Scope statement: DRP-5 reduces DQ to tractable condition, does not dissolve it |
| 2 | Regress under adversarial pressure | Sustained | Amended termination criterion: practical, not foundationalist |
| 3 | Independent falsifiability of C-DRP5.5 | Partially sustained | Joint/independent structure stated explicitly |
| 4 | Core adversarial: regress deferred? | Partially sustained | Honest scope limitation: practical resolution of DQ attribution, not epistemic regress |
| 5 | Spectral sub-protocol circularity | Dismissed | Hierarchy ordering is the DQ defence; noted for prose |

**5 challenges raised, 5 resolved. 0 unresolved.**

**HR-D5 VERDICT: PASS — proceed to Phase 3.**

---

*Hostile Review produced by Claude Code, Session 57.*
