# Bayesian Layer-2 Calibration — Research Notes

**Status:** research / pre-prototype.
**Author context:** companion to `docs/l3_meta_model_research.md` and the Betfair
closing-line study (`current_development/betfair_closing_line/RESULTS_REPORT.md`).
**Goal:** replace / augment the deterministic GLM calibrator with a *Bayesian* calibrator,
one model per walk-forward fold, that (a) corrects systematic bias, (b) **propagates and
widens** the L1 posterior so Kelly staking stops over-betting, and (c) adapts to recent form
via exponential time-decay — cheaply.

---

## 0. What the current L2 actually does (and its one structural defect)

Two shift models exist today, both on the **logit scale** with the L1 logit as a GLM `offset`:

- `BasicLogitShift`:  `logit(p_cal) = logit(p_raw) + c`,  `c` a single scalar (intercept-only
  logistic regression). This is your "affine shift that was sort of OK."
- `TeamBiasLogitShift`:  `logit(p_cal) = logit(p_raw) + c_base + β_home + β_away`, with
  `β_team` fit as **unpooled dummy variables** (one team dropped for the baseline). This is
  *already* a frequentist version of the hierarchical-team idea — but with **no shrinkage**,
  no time-decay, and a point estimate.

Both apply the shift identically to every MCMC draw:

```julia
shifted_dists = map(dist) do d
    logistic.(logit.(clamp.(d, eps, 1-eps)) .+ row_shift)   # row_shift is a SCALAR
end
```

**The structural defect.** `row_shift` is a constant, so on the logit scale every draw is
*translated by the same amount*. The shape and **width** of the L1 posterior are unchanged:

$$\mathrm{logit}(p_{\text{cal}}^{(k)}) = \mathrm{logit}(p_{\text{raw}}^{(k)}) + c
\;\;\Rightarrow\;\; \mathrm{Var}\big[\mathrm{logit}\,p_{\text{cal}}\big] = \mathrm{Var}\big[\mathrm{logit}\,p_{\text{raw}}\big].$$

The Betfair PIT result (`RESULTS_REPORT.md` §4: nominal 50/80/95% → empirical 37/64/87%,
KS D=0.090, reject uniform) says the L1 posterior is **over-confident — intervals too narrow**.
A translation *cannot fix dispersion*. **Only a calibrator whose shift carries its own variance
can widen the posterior.** That is the entire mathematical motivation for going Bayesian here.

---

## 1. The central question: "shift a distribution by a distribution → is that 2D?"

Short answer: **No. It is a 1-D convolution.** The 2-D object you're picturing is only the
*Monte-Carlo sampling grid*; the inferential target collapses to one distribution over [0,1].

### 1.1 Set-up

Fix one market `m` (e.g. `under_25`) and one match `i`. Work on the logit scale,
$z = \mathrm{logit}(p)$.

- **L1 posterior** for the match: draws $\{z_i^{(1)},\dots,z_i^{(N)}\}$ from $\pi_1(z\mid \text{match}_i)$
  ($N\approx 4000$). Carries the model's *epistemic* uncertainty about the true log-odds.
- **L2 shift posterior** (the new bit): the shift is itself random,
  $\delta \sim \pi_2(\delta \mid \text{calibration data})$, draws $\{\delta^{(1)},\dots,\delta^{(M)}\}$.

The calibrated log-odds is the deterministic map $Z_{\text{cal}} = Z + \Delta$.

### 1.2 The collapse (why it is 1-D)

"For every point in the shift distribution, shift the L1 distribution" generates the
$N\times M$ array $\{z_i^{(k)} + \delta^{(j)}\}$. But the object you *want* is a single
calibrated posterior $q_i(z_{\text{cal}})$, obtained by **marginalizing out both sources of
uncertainty**. With $Z \perp \Delta$ (justified in §1.3) this marginal is exactly the
**convolution of the two densities**:

$$
\boxed{\,q_i(z_{\text{cal}}) = \int f_{Z}(z_{\text{cal}} - \delta)\, f_{\Delta}(\delta)\, d\delta
   = (f_Z * f_\Delta)(z_{\text{cal}})\,}
$$

Convolution of two 1-D densities is a 1-D density. The $N\times M$ grid is just an i.i.d.
sample of size $NM$ **from that convolution** — flatten it and you have draws of the
calibrated posterior. You do **not** keep a 2-D array; you collapse it.

> **Practical note.** You don't need all $NM$ draws. Randomly pair indices $(k,j)$ i.i.d. and
> form $z_i^{(k)} + \delta^{(k)}$ — this is an unbiased sample from the same convolution and
> keeps the pipeline at $N$ draws (drop-in replacement for the current `shifted_dists`).

### 1.3 The one assumption that makes it clean: independence

The convolution is valid because $\pi_1$ (estimated from match/goal data by the L1 sampler)
and $\pi_2$ (estimated from a **separate, strictly earlier, walk-forward-OOS** calibration set)
are independent sources of epistemic uncertainty. Your existing architecture already enforces
this: L2 is fit only on past L1 OOS predictions (`min_history_splits`, rolling/expanding window
in `CalibrationConfig`). If you ever fit L2 on the *same* matches L1 saw, the independence —
and the leakage guarantee — breaks.

### 1.4 The payoff: variances add

For a **pure additive** Bayesian shift ($\beta=1$),

$$\mathrm{Var}[Z_{\text{cal}}] = \mathrm{Var}[Z] + \mathrm{Var}[\Delta].$$

So the Bayesian shift **widens** every match's posterior by $\mathrm{Var}[\Delta]$ on the logit
scale. That is the direct, dialable cure for the PIT over-confidence. The current deterministic
L2 has $\mathrm{Var}[\Delta]=0$ and is the $\sigma_\delta\to 0$ limit of the Bayesian one —
a clean nested check.

### 1.5 The affine case (and why to defer it)

If you also learn a slope, $Z_{\text{cal}} = \alpha + \beta Z + \delta$ with $(\alpha,\beta,\delta)\sim\pi_2$,
the collapse is still 1-D (marginalize the lot) but is now a **mixture of scaled translations**,
not a pure convolution, and

$$\mathrm{Var}[Z_{\text{cal}}] = \mathbb{E}[\beta^2]\,\mathrm{Var}[Z] + \mathrm{Var}[\beta]\,\mathbb{E}[Z]^2 + \dots$$

A slope $\beta<1$ **shrinks** the L1 logit spread — this is *tempering*, which you've said was
"always bad." **Recommendation: fix $\beta=1$ (offset form) initially.** Add the additive,
variance-injecting $\delta$ for widening; leave the slope at identity until diagnostics demand it.

---

## 2. Two defects, two distinct knobs — don't conflate them

| Defect | What it is | Diagnostic | Knob that fixes it |
|---|---|---|---|
| **Location bias** | calibrated mean systematically off | OOS log-score, reliability curve | intercept $\alpha$ (and slope $\beta$) — *current L2 already does this* |
| **Dispersion (over-confidence)** | per-match posterior too narrow | PIT / KS / interval coverage | **variance of the shift** $\mathrm{Var}[\Delta]$ — *only the Bayesian L2 does this* |

Your historical L2 work attacked **location** (intercept = fine, slope/tempering = bad) and
never had a knob for **dispersion**. The Betfair study says dispersion is the live problem.
This reframes the whole project: the new value is the *variance*, not a better mean.

---

## 3. The generative model (recommended form)

For market `m`, fold ending at calibration time $T$, over historical OOS observations
$i$ (kickoff $t_i$, outcome $y_i\in\{0,1\}$, L1 mean log-odds $\hat z_i = \mathrm{logit}(\bar p_i^{L1})$,
teams $h(i),a(i)$):

**Linear predictor (offset form, $\beta\equiv 1$):**
$$\eta_i = \hat z_i + \alpha_m + u_{h(i)} + u_{a(i)}$$

**Time-decayed (power) likelihood:**
$$\mathcal{L} = \prod_i \mathrm{Bernoulli}\big(y_i \mid \sigma(\eta_i)\big)^{\,w_i},
\qquad w_i = \exp\!\big(-c\,(T - t_i)\big)$$

**Partial pooling on team residual-bias effects (the Bayesian upgrade over the GLM):**
$$u_j \sim \mathcal{N}(0,\tau_m^2)\quad\text{(soft sum-to-zero)},\qquad
\alpha_m \sim \mathcal{N}(0, s_\alpha^2),\qquad
\tau_m \sim \mathrm{HalfNormal}(s_\tau).$$

`c` is a fixed inverse-timescale hyperparameter (see §5). The posterior we sample is
$\pi_2(\alpha_m, \{u_j\}, \tau_m \mid \text{data})$.

**Prediction for new match $i^\*$ with teams $(h^\*,a^\*)$:** the per-match shift is
$\Delta_{i^\*} = \alpha_m + u_{h^\*} + u_{a^\*}$ with $\pi_2$-posterior draws; then collapse
$z_{\text{cal}} = z_{L1} + \Delta_{i^\*}$ as in §1.2. An **unseen team** is handled natively:
draw $u \sim \mathcal{N}(0,\tau_m^2)$ from the group predictive (vs the GLM's silent
`get(...,0.0)` — which understates uncertainty for new teams, a real issue on promoted Irish sides).

### 3.1 Your literal "Bernoulli(a), a = global + δ_home + δ_away" — the fix

As written, $a$ depends **only on team identities** and omits $\hat z_i$. That is **not a
calibrator** — it's a standalone hierarchical team model that *ignores L1*. Consequences:

1. It throws away the L1 posterior entirely (replacement, not calibration) — so there is no
   "L1 distribution to shift," and the whole §1 machinery doesn't apply.
2. It **double-counts**: L1 already encodes team strength via latent attack/defence. Team-only
   deltas re-learn — with far less data — what L1 already knows, and the two layers will fight.

**Fix = keep the L1 log-odds as an offset (the $\hat z_i$ term above).** Then $\alpha_m + u_h + u_a$
means *"where is L1 systematically miscalibrated for this team"* — residual bias, **not** team
strength. If L1 is good these $u_j$ should be ≈0; large $u_j$ is then a useful diagnostic, not
the main signal.

---

## 4. Time-decay = exponentially-weighted (power / generalized) likelihood

$w_i = e^{-c\,\Delta T_i}$ raises each likelihood factor to a power → a **power posterior**
("safe Bayes" / generalized Bayes; Grünwald & van Ommen 2017; Bissiri–Holmes–Walker 2016). It
is the West–Harrison **discount factor** / exponential forgetting in likelihood form. Notes:

- It is a *pseudo*-posterior, not a coherent Bayesian update (the implied DGP changes), so
  **validate by OOS predictive score, never by Bayes factors.** This is standard and fine.
- **Effective sample size:** $\mathrm{ESS} = (\sum_i w_i)^2 / \sum_i w_i^2$. With binary
  outcomes, partial pooling, *and* decay, the binding constraint is **per-team ESS** on a thin
  league. Monitor it; it's the single most likely failure mode (§6).
- **Choosing $c$:** tie it to the L1 half-life for consistency — `DCMH_HalfLife_60`
  ⇒ half-life 60 days ⇒ $c = \ln 2 / 60 \approx 0.0116\,\text{day}^{-1}$ — *or* tune $c$ on a 1-D
  grid by maximizing OOS weighted log-score. $c$ and $\tau_m$ both control adaptivity
  (forget vs pool); identify them jointly on OOS score, don't free both blindly.

---

## 5. Computation — keep it cheap (Laplace first)

A Bernoulli-logit hierarchical model has no closed form, but it is **tiny** (per market, per
fold: one $\alpha$, a handful of $u_j$, one $\tau$). Three tiers, in order of recommendation:

1. **Laplace / MAP + Gaussian posterior** — fit a Bayesian-ridge logistic (the $u_j\sim N(0,\tau^2)$
   prior *is* an L2 ridge penalty on the team dummies; this is literally `TeamBiasLogitShift`
   **plus a ridge plus a posterior covariance**). The posterior on $\eta$ is then Gaussian, so
   the per-match shift is $\Delta_{i^\*}\sim\mathcal{N}(\mu_{i^\*}, \sigma_{i^\*}^2)$ and the
   "distributional shift" is an **analytic Gaussian convolution on the logit scale**:
   $z_{\text{cal}} = z_{L1} + \mu_{i^\*} + \sigma_{i^\*}\,\varepsilon$. Milliseconds per fit,
   one interpretable $\sigma$ for the widening. **This is the right first implementation.**
2. **ADVI** — fast mean-field if you want a quick non-Gaussian-ish posterior without NUTS.
3. **NUTS** (Turing) — only if tails/curvature matter; fits the existing `QueuedNUTSConfig`
   infra and is embarrassingly parallel across (market × fold), but almost certainly overkill.

The Laplace route makes the whole thing a per-match Gaussian-on-logit — trivial to wire into the
existing `apply_calibration` return contract `(shifted_scalars, shifted_dists)`.

---

## 6. Will it work? Is it a good idea? — honest assessment

**Verdict.** The **distributional (variance-adding) part is clearly worth doing** and is the
theoretically correct fix for the documented PIT over-confidence — the current deterministic
shift provably can't address dispersion. The **hierarchical-team part is plausible but
higher-risk** on a thin league and partly redundant with L1; add it *second*, gated on
diagnostics. The **time-decay is cheap and sound** as generalized Bayes.

**Risks, ranked:**

1. **Per-team ESS / sparsity (biggest).** Binary outcomes, one market, one fold, times decay —
   you may estimate $u_j$ from 2–5 effective games. Partial pooling mitigates (that's its job),
   but check ESS per team and be ready to pool harder (smaller $\tau$ prior) or drop team effects.
2. **Double-counting team strength** already in L1 → use the **offset form** (§3.1); interpret
   $u_j$ as residual bias; expect ≈0.
3. **Tempering history** → keep $\beta=1$; widen via $\mathrm{Var}[\Delta]$, not via slope.
4. **Pseudo-likelihood incoherence** → judge by OOS score, not Bayes factors.
5. **Leakage** → fit only on L1 OOS rows strictly before the fold (architecture already enforces).
6. **Over-widening** → if $\mathrm{Var}[\Delta]$ over-corrects, PIT flips to under-confident.
   Tune to coverage; it's a one-parameter dial.

**Why it's a good idea regardless of the team part:** even the *minimal* model — offset +
global $\alpha$ + a single posterior $\sigma$ on the shift — is a strictly better object than
the current L2 for Kelly, because Kelly's shrinkage depends on posterior *variance*, and you'd
finally be feeding it an honestly-dispersed one.

---

## 7. Evaluation protocol (pre-register before coding)

**Baselines:** raw L1 · `BasicLogitShift` · `TeamBiasLogitShift`.

**Strict walk-forward OOS metrics:**
- **Location:** OOS (weighted) log-score & Brier — must not regress vs current L2.
- **Dispersion (headline):** PIT KS-D and 50/80/95% interval coverage — must move toward
  nominal. Reuse the `pit_calibration` machinery from `l01_clv_eval.jl`.
- **Downstream:** `hurdle_G` / `hurdle_G_emp` and ROI from `run_backtest` with `BayesianKelly`
  on the calibrated posterior — does honest widening reduce over-betting and improve growth?
- **Nested sanity:** $\sigma_\delta\to 0$ must reproduce `BasicLogitShift` exactly.

**Decision rule:** ship the widening if PIT coverage improves *and* downstream `hurdle_G` does
not regress; add team effects only if they further improve OOS log-score beyond the global model.

---

## 8. Next steps (staged)

1. **Motivation baseline.** Re-run the Betfair PIT/Stage-1 on the **current** `BasicLogitShift`
   to confirm on-record that it does *not* fix dispersion (it can't — but show the number).
2. **`l0X_bayes_calib.jl` (loader) + runner.** Laplace-Gaussian Bayesian logistic, offset $=\hat z_i$,
   global $\alpha$ with $N(0,s_\alpha^2)$ prior → per-match $(\mu,\sigma)$; collapse to $N$ draws
   via $z_{L1} + \mu + \sigma\varepsilon$. One market (`under_25`), one fold. Verify the
   convolution numerically (sample $NM$ grid vs paired-$N$ vs analytic Gaussian — all agree).
3. **Add exponential weights** $w_i=e^{-c\Delta T}$; tie $c$ to the 60-day half-life; report ESS
   (global and per-team).
4. **Add partially-pooled team effects** $u_j\sim N(0,\tau^2)$; plot $\hat u_j$ vs the Irish
   "always-win" teams; gate inclusion on OOS log-score improvement.
5. **Graduate** if it wins: new `BayesianShiftCalibrator <: AbstractLayerTwoModel` implementing
   `fit_calibrator` / `apply_calibration`, returning widened `shifted_dists`. Slot into the
   existing `CalibrationConfig` rolling-window machinery unchanged.

---

## 9. References

- West & Harrison, *Bayesian Forecasting and Dynamic Models* (1997) — Ch. 6 (discount factors), Ch. 11.
- Bissiri, Holmes & Walker (2016), "A general framework for updating belief distributions," JRSS-B — generalized/Gibbs posteriors (the power-likelihood foundation).
- Grünwald & van Ommen (2017), "Inconsistency of Bayes under misspecification and the SafeBayes algorithm," *Bayesian Analysis* — why/when to temper the likelihood.
- Platt (1999), "Probabilistic outputs for SVMs…" — logistic calibration (the frequentist parent of this model).
- Kull, Silva Filho & Flach (2017), "Beta calibration" — calibration map families beyond Platt.
- Gneiting, Balabdaoui & Raftery (2007), JRSS-B — PIT calibration & sharpness (the evaluation backbone).
- Baker & McHale (2013) — Kelly under parameter uncertainty (why posterior *variance* matters downstream).
- Gelman et al., *Bayesian Data Analysis* (3e), Ch. 5 — hierarchical models / partial pooling.
