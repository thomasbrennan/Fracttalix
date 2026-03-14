# Fracttalix — Code Review & Fact-Check Assistant

I'm working on a research project called **Fracttalix** and I need your help as a **reviewer and fact-checker**. This is a mathematics/physics project making specific falsifiable claims. I need you to independently verify them.

## Project Summary

- **Project**: Fracttalix — a research corpus on the Fractal Rhythm Model (FRM)
- **Author**: Thomas Brennan
- **Licence**: CC0 public domain
- **Repo**: github.com/thomasbrennan/Fracttalix

## The Core Theory

**Claim**: A network's transient dynamics follow a damped oscillatory form:

f(t) = B + A * exp(-lambda * t) * cos(omega * t + phi)

**Key constants** (analytically derived, not curve-fitted):

| Constant | Value | Expression | Meaning |
|----------|-------|------------|---------|
| beta | 0.5 | 1/2 | Quarter-wave resonance coefficient at Hopf criticality |
| k* | 1.5708 | pi/2 | Critical feedback gain at Hopf bifurcation |
| Gamma | 3.4674 | 1 + pi^2/4 | Universal loop impedance constant |

**Scope**: Only applies where mu < 0 (damped oscillators near Hopf bifurcation).

## How to Give Me Feedback

Please structure your reviews as JSON so I can track them systematically:

```json
{
  "reviewed_claim": "F-1.1",
  "verdict": "confirmed | disputed | inconclusive | needs-revision",
  "confidence": 0.85,
  "reasoning": "Your detailed analysis...",
  "sources_checked": ["Author (Year) Title"],
  "suggestions": "Any improvements or concerns."
}
```

Be adversarial — your value is in catching errors, not confirming things. If something looks wrong, say so.

---

## First Review Batch

Here are 4 falsifiable claims from Paper 1 (The Fractal Rhythm Model: Theoretical Foundations). Please review each one independently and give me a JSON review for each.

### Claim F-1.4: beta = 1/2 Analytic Derivation (Hopf Quarter-Wave Theorem)

**Statement**: beta = 1/2 follows necessarily from the Hopf criticality condition of a normalized first-order DDE with delayed negative feedback. This is a theorem, not an empirical result.

**Derivation outline**:
1. Start with normalized DDE: dx/ds = alpha * x(s) - k * x(s - 1)
2. At criticality alpha = 0, substitute lambda = i*Omega into characteristic equation
3. Real part: k * cos(Omega) = 0, so Omega* = pi/2
4. Therefore omega* = Omega*/tau_gen = pi/(2*tau_gen), giving beta = 1/2

**Falsification predicate**: FALSIFIED IF any step in the analytic proof is invalid (does not follow from inputs by named inference rule), OR if the derivation output is not equal to 0.5.

**Context**: Hayes (1950) and Kuang (1993) establish omega*tau = pi/2 in individual domains. The novelty claim is identifying this as the universal source of empirical beta = 1/2 across multiple independent domains.

**Questions for you**:
1. Is the derivation logically valid step by step?
2. Does this correctly follow from standard DDE theory (Hayes 1950, Kuang 1993)?
3. Is the novelty claim fair — has anyone else made this specific cross-domain identification?

---

### Claim F-1.5: Lambda Derivation (Leading Order)

**Statement**: The FRM damping rate lambda = |alpha| / (Gamma * tau_gen) where Gamma = 1 + pi^2/4 = 3.467 is the universal loop impedance constant, derived from the DDE characteristic equation via perturbation expansion about the Hopf critical point.

**Falsification predicate**: FALSIFIED IF any step in the perturbation derivation is invalid, OR if mean(lambda_fit / lambda_theory) across 8 simulation points falls outside [0.85, 1.15].

**Observed**: Mean ratio 1.036 +/- 0.089 across stable regime. Gamma = |dh/dlambda|* = |1 + i*pi/2|^2 = 1 + pi^2/4.

**Questions for you**:
1. Does Gamma = 1 + pi^2/4 correctly fall out of the perturbation expansion?
2. Check: |1 + i*pi/2|^2 = 1 + pi^2/4. Is this the right squared modulus?
3. Is 3.61% mean error reasonable for a leading-order perturbation result?

---

### Claim F-1.6: Circadian Period Prediction

**Statement**: T = 4 * tau_gen predicts the mammalian circadian period T = 24 hr from tau_gen = 6 hr with no fitted parameters. tau_gen is independently measured by molecular biology sources.

**Falsification predicate**: FALSIFIED IF T_predicted < 20 hr OR T_predicted > 28 hr, OR if fewer than 3 independent sources confirm tau_gen in the 5.5-6.5 hr range.

**Sources cited**:
- Kim & Forger 2012: tau_gen = 5.9 hr (direct measurement)
- Hardin et al. 1990: ~6 hr
- Lee et al. 2001: ~6 hr sub-step
- Takahashi 2017: ~6 hr

**Questions for you**:
1. Can you verify these citations? Do Kim & Forger (2012), Hardin et al. (1990), Lee et al. (2001), and Takahashi (2017) actually report these tau_gen values?
2. Is T = 4 * tau_gen a novel prediction, or has this relationship been noted before?
3. Are there contradictory measurements of tau_gen that would challenge the 5.5-6.5 hr range?

---

### Claim F-1.7: Stuart-Landau Connection

**Statement**: FRM is the approximate transient solution of the Stuart-Landau normal form in the linear regime for mu < 0, with lambda ≈ |mu|. Nonlinear terms introduce additional damping (observed slope k=1.10). The FRM scope boundary coincides with the Hopf bifurcation.

**Falsification predicate**: FALSIFIED IF max pointwise residual between FRM fit and Stuart-Landau ODE solution exceeds 0.05, OR if R^2 < 0.99 for any mu in {-3.0, -2.0, -1.0, -0.5, -0.3, -0.1}.

**Observed**: R^2 > 0.99 confirmed for mu in {-3.0, -1.0}. lambda_fit tracks |mu| with slope k = 1.10, R^2 = 0.965.

**Questions for you**:
1. Is it mathematically correct that the Stuart-Landau normal form dz/dt = (mu + i*omega_0)*z - |z|^2*z yields damped oscillatory transients of the form B + A*exp(-lambda*t)*cos(omega*t + phi) for mu < 0?
2. The slope k = 1.10 instead of exactly 1.0 — is this expected or a red flag?
3. Does the scope boundary (mu = 0 as Hopf bifurcation) correctly mark where FRM breaks down?
